
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Precomputed unit vectors for 9 moves
DIRECTIONS = {
    "STAY": (0.0, 0.0),
    "UP": (0.0, 1.0),
    "UP_RIGHT": (0.7071067811865475, 0.7071067811865475),
    "RIGHT": (1.0, 0.0),
    "DOWN_RIGHT": (0.7071067811865475, -0.7071067811865475),
    "DOWN": (0.0, -1.0),
    "DOWN_LEFT": (-0.7071067811865475, -0.7071067811865475),
    "LEFT": (-1.0, 0.0),
    "UP_LEFT": (-0.7071067811865475, 0.7071067811865475),
}
PLAYERS = ("player_1", "player_2")


def _hypot(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(v[0], v[1])
    return (0.0, 0.0) if length == 0.0 else (v[0] / length, v[1] / length)


@dataclass
class FastObstacle:
    x: float
    y: float
    width: float
    height: float


@dataclass
class FastConfig:
    field_width: float = 100.0
    field_height: float = 140.0
    goal_width: float = 36.0
    player_radius: float = 3.0
    player_speed: float = 4.0
    ball_radius: float = 1.5
    ball_speed: float = 8.0
    possession_radius: float = 5.0
    kick_distances: tuple[float, float, float] = (32.0, 64.0, 96.0)
    possession_limit_iterations: int = 10
    loose_ball_restart_iterations: int = 20
    maximum_iterations: int = 400
    maximum_goals: int = 7
    initial_possessor: str = "player_1"


class FastState:
    __slots__ = (
        "p1_x", "p1_y", "p2_x", "p2_y",
        "ball_x", "ball_y", "ball_vx", "ball_vy",
        "ball_rem_dist", "possession", "possession_steps",
        "loose_ball_steps", "loose_ball_best_dist", "last_touch",
        "score_1", "score_2", "iteration", "done", "termination_reason"
    )

    def __init__(
        self,
        p1_x: float = 50.0, p1_y: float = 35.0,
        p2_x: float = 50.0, p2_y: float = 105.0,
        ball_x: float = 50.0, ball_y: float = 35.0,
        ball_vx: float = 0.0, ball_vy: float = 0.0,
        ball_rem_dist: float = 0.0,
        possession: str | None = "player_1",
        possession_steps: int = 0,
        loose_ball_steps: int = 0,
        loose_ball_best_dist: float | None = None,
        last_touch: str | None = "player_1",
        score_1: int = 0, score_2: int = 0,
        iteration: int = 0,
        done: bool = False,
        termination_reason: str | None = None,
    ):
        self.p1_x = p1_x; self.p1_y = p1_y
        self.p2_x = p2_x; self.p2_y = p2_y
        self.ball_x = ball_x; self.ball_y = ball_y
        self.ball_vx = ball_vx; self.ball_vy = ball_vy
        self.ball_rem_dist = ball_rem_dist
        self.possession = possession
        self.possession_steps = possession_steps
        self.loose_ball_steps = loose_ball_steps
        self.loose_ball_best_dist = loose_ball_best_dist
        self.last_touch = last_touch
        self.score_1 = score_1; self.score_2 = score_2
        self.iteration = iteration
        self.done = done
        self.termination_reason = termination_reason

    def clone(self) -> FastState:
        s = FastState.__new__(FastState)
        s.p1_x = self.p1_x; s.p1_y = self.p1_y
        s.p2_x = self.p2_x; s.p2_y = self.p2_y
        s.ball_x = self.ball_x; s.ball_y = self.ball_y
        s.ball_vx = self.ball_vx; s.ball_vy = self.ball_vy
        s.ball_rem_dist = self.ball_rem_dist
        s.possession = self.possession
        s.possession_steps = self.possession_steps
        s.loose_ball_steps = self.loose_ball_steps
        s.loose_ball_best_dist = self.loose_ball_best_dist
        s.last_touch = self.last_touch
        s.score_1 = self.score_1; s.score_2 = self.score_2
        s.iteration = self.iteration
        s.done = self.done
        s.termination_reason = self.termination_reason
        return s

    def copy_into(self, target: FastState) -> None:
        target.p1_x = self.p1_x; target.p1_y = self.p1_y
        target.p2_x = self.p2_x; target.p2_y = self.p2_y
        target.ball_x = self.ball_x; target.ball_y = self.ball_y
        target.ball_vx = self.ball_vx; target.ball_vy = self.ball_vy
        target.ball_rem_dist = self.ball_rem_dist
        target.possession = self.possession
        target.possession_steps = self.possession_steps
        target.loose_ball_steps = self.loose_ball_steps
        target.loose_ball_best_dist = self.loose_ball_best_dist
        target.last_touch = self.last_touch
        target.score_1 = self.score_1; target.score_2 = self.score_2
        target.iteration = self.iteration
        target.done = self.done
        target.termination_reason = self.termination_reason


class FastSimulator:
    def __init__(self, config: FastConfig, obstacles: list[FastObstacle] | None = None):
        self.cfg = config
        self.obstacles = obstacles or []
        self.fw = config.field_width
        self.fh = config.field_height
        self.gw = config.goal_width
        self.pr = config.player_radius
        self.ps = config.player_speed
        self.br = config.ball_radius
        self.bs = config.ball_speed
        self.pos_rad = config.possession_radius
        self.goal_left = (self.fw - self.gw) / 2.0
        self.goal_right = self.goal_left + self.gw
        self.substep_lim = max(0.25, self.br * 0.45)
        self.contact_dist = 2.0 * self.pr
        self.tackle_dist = 2.0 * self.pr + 0.15

    def from_env_observation(
        self,
        obs: dict[str, Any],
        loose_ball_best_dist: float | None = None,
    ) -> FastState:
        st = obs["state"]
        p1 = st["players"]["player_1"]
        p2 = st["players"]["player_2"]
        b = st["ball"]
        sc = st.get("score", {})
        return FastState(
            p1_x=float(p1["x"]), p1_y=float(p1["y"]),
            p2_x=float(p2["x"]), p2_y=float(p2["y"]),
            ball_x=float(b["x"]), ball_y=float(b["y"]),
            ball_vx=float(b.get("velocity", {}).get("x", 0.0)),
            ball_vy=float(b.get("velocity", {}).get("y", 0.0)),
            ball_rem_dist=float(b.get("remaining_kick_distance", 0.0)),
            possession=b.get("possession"),
            possession_steps=int(b.get("possession_steps", 0)),
            loose_ball_steps=int(b.get("loose_ball_steps", 0)),
            loose_ball_best_dist=loose_ball_best_dist,
            last_touch=b.get("possession") or "player_1",
            score_1=int(sc.get("player_1", 0)),
            score_2=int(sc.get("player_2", 0)),
            iteration=int(st.get("iteration", 0)),
            done=bool(st.get("done", False)),
            termination_reason=st.get("termination_reason")
        )

    def is_valid_player_pos(self, x: float, y: float) -> bool:
        if x - self.pr < 0.0 or x + self.pr > self.fw:
            return False
        if y - self.pr < 0.0 or y + self.pr > self.fh:
            return False
        for obs in self.obstacles:
            cx = min(max(x, obs.x), obs.x + obs.width)
            cy = min(max(y, obs.y), obs.y + obs.height)
            if math.hypot(x - cx, y - cy) < self.pr:
                return False
        return True

    def step(
        self,
        st: FastState,
        p1_move: str,
        p2_move: str,
        p1_kick: tuple[str, int] | None = None,
        p2_kick: tuple[str, int] | None = None,
    ) -> None:
        if st.done:
            return

        # 1. Move players
        old_p1_x, old_p1_y = st.p1_x, st.p1_y
        old_p2_x, old_p2_y = st.p2_x, st.p2_y

        u1 = DIRECTIONS[p1_move]
        u2 = DIRECTIONS[p2_move]

        cand1_x = old_p1_x + u1[0] * self.ps
        cand1_y = old_p1_y + u1[1] * self.ps
        prop1_x, prop1_y = (cand1_x, cand1_y) if self.is_valid_player_pos(cand1_x, cand1_y) else (old_p1_x, old_p1_y)

        cand2_x = old_p2_x + u2[0] * self.ps
        cand2_y = old_p2_y + u2[1] * self.ps
        prop2_x, prop2_y = (cand2_x, cand2_y) if self.is_valid_player_pos(cand2_x, cand2_y) else (old_p2_x, old_p2_y)

        dist_prop = math.hypot(prop1_x - prop2_x, prop1_y - prop2_y)
        if dist_prop < self.contact_dist:
            # Contact resolution
            mid_x = (prop1_x + prop2_x) * 0.5
            mid_y = (prop1_y + prop2_y) * 0.5
            sep = _unit((old_p1_x - old_p2_x, old_p1_y - old_p2_y))
            if sep == (0.0, 0.0):
                sep = (1.0, 0.0)

            res1_x = mid_x + sep[0] * self.pr
            res1_y = mid_y + sep[1] * self.pr
            res2_x = mid_x - sep[0] * self.pr
            res2_y = mid_y - sep[1] * self.pr

            resolved_movement = math.hypot(old_p1_x - res1_x, old_p1_y - res1_y) + math.hypot(old_p2_x - res2_x, old_p2_y - res2_y)
            if resolved_movement > 0.1 and self.is_valid_player_pos(res1_x, res1_y) and self.is_valid_player_pos(res2_x, res2_y):
                prop1_x, prop1_y = res1_x, res1_y
                prop2_x, prop2_y = res2_x, res2_y
            else:
                # Solo mover rule by ball progress
                ball_x, ball_y = st.ball_x, st.ball_y
                solo_moves: list[tuple[float, str]] = []

                if self.is_valid_player_pos(prop1_x, prop1_y) and math.hypot(prop1_x - old_p2_x, prop1_y - old_p2_y) >= self.contact_dist:
                    prog1 = math.hypot(old_p1_x - ball_x, old_p1_y - ball_y) - math.hypot(prop1_x - ball_x, prop1_y - ball_y)
                    solo_moves.append((prog1, "player_1"))

                if self.is_valid_player_pos(prop2_x, prop2_y) and math.hypot(prop2_x - old_p1_x, prop2_y - old_p1_y) >= self.contact_dist:
                    prog2 = math.hypot(old_p2_x - ball_x, old_p2_y - ball_y) - math.hypot(prop2_x - ball_x, prop2_y - ball_y)
                    solo_moves.append((prog2, "player_2"))

                if solo_moves:
                    _, mover = max(solo_moves, key=lambda item: (item[0], item[1]))
                    if mover == "player_1":
                        cand_x = old_p1_x + u1[0] * self.ps
                        cand_y = old_p1_y + u1[1] * self.ps
                        prop1_x, prop1_y = (cand_x, cand_y) if self.is_valid_player_pos(cand_x, cand_y) else (old_p1_x, old_p1_y)
                        prop2_x, prop2_y = old_p2_x, old_p2_y
                    else:
                        cand_x = old_p2_x + u2[0] * self.ps
                        cand_y = old_p2_y + u2[1] * self.ps
                        prop2_x, prop2_y = (cand_x, cand_y) if self.is_valid_player_pos(cand_x, cand_y) else (old_p2_x, old_p2_y)
                        prop1_x, prop1_y = old_p1_x, old_p1_y
                else:
                    # Sidestep alternatives
                    alts: list[tuple[float, str, float, float]] = []
                    for m_name, vec in DIRECTIONS.items():
                        if m_name == "STAY":
                            continue
                        su = vec
                        c1x = old_p1_x + su[0] * self.ps
                        c1y = old_p1_y + su[1] * self.ps
                        if self.is_valid_player_pos(c1x, c1y) and math.hypot(c1x - old_p2_x, c1y - old_p2_y) >= self.contact_dist:
                            p1_prg = math.hypot(old_p1_x - ball_x, old_p1_y - ball_y) - math.hypot(c1x - ball_x, c1y - ball_y)
                            alts.append((p1_prg, "player_1", c1x, c1y))

                        c2x = old_p2_x + su[0] * self.ps
                        c2y = old_p2_y + su[1] * self.ps
                        if self.is_valid_player_pos(c2x, c2y) and math.hypot(c2x - old_p1_x, c2y - old_p1_y) >= self.contact_dist:
                            p2_prg = math.hypot(old_p2_x - ball_x, old_p2_y - ball_y) - math.hypot(c2x - ball_x, c2y - ball_y)
                            alts.append((p2_prg, "player_2", c2x, c2y))

                    if alts:
                        _, s_mover, sc_x, sc_y = max(alts, key=lambda item: (item[0], item[1], item[2]))
                        if s_mover == "player_1":
                            prop1_x, prop1_y = sc_x, sc_y
                            prop2_x, prop2_y = old_p2_x, old_p2_y
                        else:
                            prop1_x, prop1_y = old_p1_x, old_p1_y
                            prop2_x, prop2_y = sc_x, sc_y
                    else:
                        prop1_x, prop1_y = old_p1_x, old_p1_y
                        prop2_x, prop2_y = old_p2_x, old_p2_y

        st.p1_x, st.p1_y = prop1_x, prop1_y
        st.p2_x, st.p2_y = prop2_x, prop2_y

        # 2. Resolve tackle
        if st.possession is not None:
            owner = st.possession
            challenger = "player_2" if owner == "player_1" else "player_1"
            challenger_move = p2_move if challenger == "player_2" else p1_move
            owner_kick = p1_kick if owner == "player_1" else p2_kick

            if st.possession_steps >= 3 and owner_kick is None:
                if challenger_move != "STAY" and math.hypot(prop1_x - prop2_x, prop1_y - prop2_y) <= self.tackle_dist:
                    st.possession = challenger
                    st.possession_steps = 0
                    st.last_touch = challenger
                    if challenger == "player_1":
                        st.ball_x, st.ball_y = prop1_x, prop1_y
                    else:
                        st.ball_x, st.ball_y = prop2_x, prop2_y

        # 3. Kicking
        if st.possession is not None:
            curr_owner = st.possession
            st.possession_steps += 1
            if curr_owner == "player_1":
                st.ball_x, st.ball_y = st.p1_x, st.p1_y
                kick_req = p1_kick
            else:
                st.ball_x, st.ball_y = st.p2_x, st.p2_y
                kick_req = p2_kick

            if kick_req is not None:
                kdir, kpow = kick_req
                self._execute_kick(st, curr_owner, kdir, kpow)
            elif st.possession_steps >= self.cfg.possession_limit_iterations:
                auto_dir = "UP" if curr_owner == "player_1" else "DOWN"
                self._execute_kick(st, curr_owner, auto_dir, 1)

        # 4. Ball flight
        scorer = None
        if st.possession is None and (st.ball_rem_dist > 0.0 or (st.ball_vx != 0.0 or st.ball_vy != 0.0)):
            scorer = self._move_ball(st)

        # 5. Goal resolution / Stationary claim / Stalled drop ball
        if scorer:
            st.loose_ball_steps = 0
            st.loose_ball_best_dist = None
            if scorer == "player_1":
                st.score_1 += 1
                conceder = "player_2"
            else:
                st.score_2 += 1
                conceder = "player_1"

            st.iteration += 1
            if (st.score_1 + st.score_2) >= self.cfg.maximum_goals:
                st.done = True
                st.termination_reason = "maximum_goals"
            elif st.iteration >= self.cfg.maximum_iterations:
                st.done = True
                st.termination_reason = "maximum_iterations"
            else:
                self._restart_after_goal(st, conceder)
            return

        # No goal: check stationary claim & stalled drop ball
        self._claim_stationary_ball(st)
        self._restart_stalled_loose_ball(st)

        st.iteration += 1
        if (st.score_1 + st.score_2) >= self.cfg.maximum_goals:
            st.done = True
            st.termination_reason = "maximum_goals"
        elif st.iteration >= self.cfg.maximum_iterations:
            st.done = True
            st.termination_reason = "maximum_iterations"

    def _execute_kick(self, st: FastState, player: str, kdir: str, kpow: int) -> None:
        u = DIRECTIONS[kdir]
        clearance = self.pr + self.br + 0.05
        ox = st.p1_x if player == "player_1" else st.p2_x
        oy = st.p1_y if player == "player_1" else st.p2_y
        st.ball_x = ox + u[0] * clearance
        st.ball_y = oy + u[1] * clearance
        st.ball_vx = u[0] * self.bs
        st.ball_vy = u[1] * self.bs
        st.ball_rem_dist = self.cfg.kick_distances[kpow - 1]
        st.possession = None
        st.possession_steps = 0
        st.loose_ball_steps = 0
        st.loose_ball_best_dist = None
        st.last_touch = player

    def _move_ball(self, st: FastState) -> str | None:
        rem = st.ball_rem_dist
        if rem <= 0.0:
            return None

        travel = min(self.bs, rem)
        substeps = max(1, math.ceil(travel / self.substep_lim))
        step_d = travel / substeps

        bx, by = st.ball_x, st.ball_y
        bvx, bvy = st.ball_vx, st.ball_vy

        for _ in range(substeps):
            uvx, uvy = _unit((bvx, bvy))
            prev_x, prev_y = bx, by
            cand_x = prev_x + uvx * step_d
            cand_y = prev_y + uvy * step_d

            # Goal check
            if self.goal_left <= cand_x <= self.goal_right:
                if cand_y + self.br >= self.fh:
                    st.ball_x, st.ball_y = cand_x, cand_y
                    st.ball_rem_dist = 0.0
                    st.ball_vx, st.ball_vy = 0.0, 0.0
                    return "player_1"
                if cand_y - self.br <= 0.0:
                    st.ball_x, st.ball_y = cand_x, cand_y
                    st.ball_rem_dist = 0.0
                    st.ball_vx, st.ball_vy = 0.0, 0.0
                    return "player_2"

            # Wall bounces
            if cand_x - self.br < 0.0 or cand_x + self.br > self.fw:
                bvx = -bvx
                cand_x = min(max(cand_x, self.br), self.fw - self.br)
            if cand_y - self.br < 0.0 or cand_y + self.br > self.fh:
                bvy = -bvy
                cand_y = min(max(cand_y, self.br), self.fh - self.br)

            # Obstacle bounces
            for obs in self.obstacles:
                cx = min(max(cand_x, obs.x), obs.x + obs.width)
                cy = min(max(cand_y, obs.y), obs.y + obs.height)
                if math.hypot(cand_x - cx, cand_y - cy) < self.br:
                    cx_hit = (prev_x <= obs.x - self.br or prev_x >= obs.x + obs.width + self.br)
                    cy_hit = (prev_y <= obs.y - self.br or prev_y >= obs.y + obs.height + self.br)
                    if cx_hit:
                        bvx = -bvx
                    if cy_hit:
                        bvy = -bvy
                    if not cx_hit and not cy_hit:
                        bvx, bvy = -bvx, -bvy
                    d = _unit((bvx, bvy))
                    shift = min(0.05, self.br * 0.1)
                    cand_x = prev_x + d[0] * shift
                    cand_y = prev_y + d[1] * shift
                    break

            bx, by = cand_x, cand_y
            rem = max(0.0, rem - step_d)

            # In-flight interception
            d1 = math.hypot(bx - st.p1_x, by - st.p1_y)
            d2 = math.hypot(bx - st.p2_x, by - st.p2_y)
            catch_dist = self.br + self.pr

            if d1 <= catch_dist:
                st.possession = "player_1"
                st.possession_steps = 0
                st.loose_ball_steps = 0
                st.loose_ball_best_dist = None
                st.last_touch = "player_1"
                st.ball_x, st.ball_y = st.p1_x, st.p1_y
                st.ball_vx, st.ball_vy = 0.0, 0.0
                st.ball_rem_dist = 0.0
                return None
            elif d2 <= catch_dist:
                st.possession = "player_2"
                st.possession_steps = 0
                st.loose_ball_steps = 0
                st.loose_ball_best_dist = None
                st.last_touch = "player_2"
                st.ball_x, st.ball_y = st.p2_x, st.p2_y
                st.ball_vx, st.ball_vy = 0.0, 0.0
                st.ball_rem_dist = 0.0
                return None

        st.ball_x, st.ball_y = bx, by
        st.ball_vx, st.ball_vy = bvx, bvy
        st.ball_rem_dist = rem
        if rem <= 1e-9:
            st.ball_rem_dist = 0.0
            st.ball_vx, st.ball_vy = 0.0, 0.0
        return None

    def _claim_stationary_ball(self, st: FastState) -> None:
        if st.possession is not None or st.ball_rem_dist > 0.0:
            return
        d1 = math.hypot(st.p1_x - st.ball_x, st.p1_y - st.ball_y)
        d2 = math.hypot(st.p2_x - st.ball_x, st.p2_y - st.ball_y)

        cands: list[tuple[float, str]] = []
        if d1 <= self.pos_rad:
            cands.append((d1, "player_1"))
        if d2 <= self.pos_rad:
            cands.append((d2, "player_2"))

        if cands:
            _, winner = min(cands, key=lambda item: (item[0], item[1]))
            st.possession = winner
            st.possession_steps = 0
            st.loose_ball_steps = 0
            st.loose_ball_best_dist = None
            st.last_touch = winner
            st.ball_x = st.p1_x if winner == "player_1" else st.p2_x
            st.ball_y = st.p1_y if winner == "player_1" else st.p2_y

    def _restart_stalled_loose_ball(self, st: FastState) -> None:
        if st.possession is not None or st.ball_rem_dist > 0.0:
            st.loose_ball_steps = 0
            st.loose_ball_best_dist = None
            return

        closest_d = min(
            math.hypot(st.p1_x - st.ball_x, st.p1_y - st.ball_y),
            math.hypot(st.p2_x - st.ball_x, st.p2_y - st.ball_y),
        )
        if st.loose_ball_best_dist is None or closest_d < st.loose_ball_best_dist - 0.25:
            st.loose_ball_best_dist = closest_d
            st.loose_ball_steps = 0
            return

        st.loose_ball_steps += 1
        if st.loose_ball_steps >= self.cfg.loose_ball_restart_iterations:
            st.ball_x = self.fw * 0.5
            st.ball_y = self.fh * 0.5
            st.ball_vx, st.ball_vy = 0.0, 0.0
            st.ball_rem_dist = 0.0
            st.loose_ball_steps = 0
            st.loose_ball_best_dist = None

    def _restart_after_goal(self, st: FastState, conceder: str) -> None:
        st.p1_x = self.fw * 0.5
        st.p1_y = self.fh * 0.25
        st.p2_x = self.fw * 0.5
        st.p2_y = self.fh * 0.75
        st.possession = conceder
        st.possession_steps = 0
        st.loose_ball_steps = 0
        st.loose_ball_best_dist = None
        st.ball_x = st.p1_x if conceder == "player_1" else st.p2_x
        st.ball_y = st.p1_y if conceder == "player_1" else st.p2_y
        st.ball_vx, st.ball_vy = 0.0, 0.0
        st.ball_rem_dist = 0.0
        st.last_touch = conceder
