
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .simulator import (
        FastConfig, FastObstacle, FastSimulator, FastState
    )
    from .search import DefensiveSearch
    from .shot_planner import ShotPlanner
except (ImportError, ValueError):
    try:
        from my_team.team_bot.simulator import (
            FastConfig, FastObstacle, FastSimulator, FastState
        )
        from my_team.team_bot.search import DefensiveSearch
        from my_team.team_bot.shot_planner import ShotPlanner
    except ImportError:
        from simulator import (
            FastConfig, FastObstacle, FastSimulator, FastState
        )
        from search import DefensiveSearch
        from shot_planner import ShotPlanner

# ── Geometric Primitives & Simulation Configuration ──────────────────────

DIRECTIONS: dict[str, tuple[float, float]] = {
    "STAY": (0.0, 0.0),
    "UP": (0.0, 1.0),
    "UP_RIGHT": (1.0, 1.0),
    "RIGHT": (1.0, 0.0),
    "DOWN_RIGHT": (1.0, -1.0),
    "DOWN": (0.0, -1.0),
    "DOWN_LEFT": (-1.0, -1.0),
    "LEFT": (-1.0, 0.0),
    "UP_LEFT": (-1.0, 1.0),
}
MOVES: list[str] = list(DIRECTIONS)
KICK_DIRS: list[str] = [d for d in MOVES if d != "STAY"]
PLAYERS = ("player_1", "player_2")


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(*vector)
    return (0.0, 0.0) if length == 0 else (vector[0] / length, vector[1] / length)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Rectangle:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class SimConfig:
    field_width: float = 100.0
    field_height: float = 140.0
    goal_width: float = 36.0
    player_radius: float = 3.0
    player_speed: float = 4.0
    ball_radius: float = 1.5
    ball_speed: float = 8.0
    possession_radius: float = 5.0
    kick_distances: tuple[float, ...] = (32.0, 64.0, 96.0)
    possession_limit_iterations: int = 10
    loose_ball_timeout_iterations: int = 20
    maximum_iterations: int = 400
    maximum_goals: int = 7
    initial_possessor: str = "player_1"


def simulate_trajectory(
    start_x: float,
    start_y: float,
    vx: float,
    vy: float,
    rem_dist: float,
    field_w: float,
    field_h: float,
    goal_left: float,
    goal_right: float,
    ball_radius: float,
    ball_speed: float,
    sign: int,
    obstacles: list[Rectangle],
    max_steps: int = 15,
) -> tuple[bool, bool, list[tuple[float, float, int]]]:
    """
    Simulates ball flight across up to max_steps steps matching engine substep physics.
    Returns (scored, conceded, trajectory) where trajectory is [(x, y, step), ...].
    """
    substep_lim = max(0.25, ball_radius * 0.45)
    b_cur_x = start_x
    b_cur_y = start_y
    b_vx = vx
    b_vy = vy
    rem = rem_dist
    trajectory: list[tuple[float, float, int]] = []
    scored = False
    conceded = False
    step_count = 0

    while rem > 1e-9 and not scored and not conceded:
        step_count += 1
        if step_count > max_steps:
            break

        travel_step = min(ball_speed, rem)
        substeps = max(1, math.ceil(travel_step / substep_lim))
        step_d = travel_step / substeps

        for _ in range(substeps):
            uvx, uvy = _unit((b_vx, b_vy))
            prev_x, prev_y = b_cur_x, b_cur_y
            b_cur_x += uvx * step_d
            b_cur_y += uvy * step_d

            in_mouth = goal_left <= b_cur_x <= goal_right
            if in_mouth and (b_cur_y + ball_radius >= field_h if sign > 0 else b_cur_y - ball_radius <= 0.0):
                scored = True
                break
            if in_mouth and (b_cur_y - ball_radius <= 0.0 if sign > 0 else b_cur_y + ball_radius >= field_h):
                conceded = True
                break

            # Wall bounces
            if b_cur_x - ball_radius < 0:
                b_vx = abs(b_vx); b_cur_x = ball_radius
            elif b_cur_x + ball_radius > field_w:
                b_vx = -abs(b_vx); b_cur_x = field_w - ball_radius
            if b_cur_y - ball_radius < 0 and not (goal_left <= b_cur_x <= goal_right):
                b_vy = abs(b_vy); b_cur_y = ball_radius
            elif b_cur_y + ball_radius > field_h and not (goal_left <= b_cur_x <= goal_right):
                b_vy = -abs(b_vy); b_cur_y = field_h - ball_radius

            # Obstacle bounces
            for obs in obstacles:
                cx = min(max(b_cur_x, obs.x), obs.x + obs.width)
                cy = min(max(b_cur_y, obs.y), obs.y + obs.height)
                if math.hypot(b_cur_x - cx, b_cur_y - cy) < ball_radius:
                    cx_hit = (prev_x <= obs.x - ball_radius or prev_x >= obs.x + obs.width + ball_radius)
                    cy_hit = (prev_y <= obs.y - ball_radius or prev_y >= obs.y + obs.height + ball_radius)
                    if cx_hit: b_vx = -b_vx
                    if cy_hit: b_vy = -b_vy
                    if not cx_hit and not cy_hit: b_vx, b_vy = -b_vx, -b_vy
                    d = _unit((b_vx, b_vy))
                    b_cur_x = prev_x + d[0] * min(0.05, ball_radius / 10)
                    b_cur_y = prev_y + d[1] * min(0.05, ball_radius / 10)
                    break

        rem = max(0.0, rem - travel_step)
        trajectory.append((b_cur_x, b_cur_y, step_count))

    return scored, conceded, trajectory


def _direction_toward(dx: float, dy: float, dead_zone: float = 0.5) -> str:
    h = "" if abs(dx) <= dead_zone else ("RIGHT" if dx > 0 else "LEFT")
    v = "" if abs(dy) <= dead_zone else ("UP" if dy > 0 else "DOWN")
    return f"{v}_{h}" if (v and h) else (v or h or "STAY")


def _is_move_safe(
    x: float,
    y: float,
    move: str,
    field_w: float,
    field_h: float,
    player_radius: float,
    player_speed: float,
    obstacles: list[Rectangle],
) -> bool:
    if move == "STAY":
        return True
    vx, vy = DIRECTIONS[move]
    ux, uy = _unit((vx, vy))
    nx = x + ux * player_speed
    ny = y + uy * player_speed

    if nx - player_radius < 0 or nx + player_radius > field_w:
        return False
    if ny - player_radius < 0 or ny + player_radius > field_h:
        return False

    # Tangential obstacle clearance: tightened to 0.08 for maximum corridor maneuverability
    for obs in obstacles:
        cx = min(max(nx, obs.x), obs.x + obs.width)
        cy = min(max(ny, obs.y), obs.y + obs.height)
        if math.hypot(nx - cx, ny - cy) < player_radius + 0.08:
            return False
    return True


def _best_safe_move(
    x: float,
    y: float,
    preferred: str,
    field_w: float,
    field_h: float,
    player_radius: float,
    player_speed: float,
    obstacles: list[Rectangle],
) -> str:
    pv = DIRECTIONS[preferred]
    choices = [
        m
        for m in MOVES
        if m != "STAY"
        and _is_move_safe(
            x, y, m, field_w, field_h, player_radius, player_speed, obstacles
        )
    ]
    if not choices:
        return "STAY"
    return max(
        choices,
        key=lambda m: (
            DIRECTIONS[m][0] * pv[0] + DIRECTIONS[m][1] * pv[1],
            m == preferred,
        ),
    )


def _ray_hits_circle(
    px: float, py: float,
    ux: float, uy: float,
    max_dist: float,
    cx: float, cy: float,
    radius: float,
) -> bool:
    """Returns True iff ray from (px, py) in unit dir (ux, uy) intersects circle (cx, cy, radius)."""
    dx = cx - px
    dy = cy - py
    s = dx * ux + dy * uy
    if s < 0 or s > max_dist:
        return False
    d_perp = abs(dx * uy - dy * ux)
    return d_perp <= radius


_ERROR_COUNT = 0

def _record_error(err: Exception) -> None:
    global _ERROR_COUNT
    _ERROR_COUNT += 1
    if _ERROR_COUNT % 50 == 1:
        tb = err.__traceback__
        lineno = tb.tb_lineno if tb else "?"
        sys.stderr.write(
            f"Policy handled exception [{type(err).__name__}] at line {lineno} (count={_ERROR_COUNT}): {err}\n"
        )
        sys.stderr.flush()


class Policy:

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.config = SimConfig()
        self.params: dict[str, Any] = {
            "goal_clearance_margin": 0.5,
            "risky_shot_margin_cutoff": -2.0,
            "max_direct_shot_distance": 44.0,
            "safe_dribble_opp_dist": 7.0,
            "min_pass_lead_steps": 1.5,
            "midfield_press_threshold": 42.0,
            "midfield_press_cushion_max": 10.0,
            "midfield_press_cushion_min": 7.0,
            "flag_tackle_plus_kick": True,
            "flag_defensive_search": True,
            "flag_shot_planner": True,
        }
        if params:
            self.params.update(params)
        self._sig: dict = {}
        self._last_iter = -1
        self._px = -999.0
        self._py = -999.0
        self._stuck = 0
        self._last_best = None
        self._dstuck = 0
        self._history: list[tuple[float, float]] = []
        self._loose_ball_best_dist: float | None = None
        self._fast_sim: FastSimulator | None = None
        self._def_searcher: DefensiveSearch | None = None
        self._last_obs_obstacles: list | None = None
        self._exact_hold_pos: tuple[float, float] | None = None
        self._exact_hold_steps: int = 0
        self._last_wall_kick: tuple[str, int] | None = None
        self._wall_kick_repeats: int = 0

    def _sigkey(self, act, mx, my):
        k = act.get("kick")
        if not k:
            return None
        return (k.get("direction"), k.get("power"), int(mx // 10), int(my // 10))

    def _sig_n(self, act, mx, my):
        key = self._sigkey(act, mx, my)
        return self._sig.get(key, 0) if key else 0

    def _sig_mark(self, act, mx, my):
        key = self._sigkey(act, mx, my)
        if key:
            self._sig[key] = self._sig.get(key, 0) + 1
        return act

    def _is_in_periodic_cycle(self) -> bool:
        n = len(self._history)
        if (
            n >= 4
            and self._history[-1] == self._history[-3]
            and self._history[-2] == self._history[-4]
            and self._history[-1] != self._history[-2]
        ):
            return True
        if (
            n >= 6
            and self._history[-1] == self._history[-4]
            and self._history[-2] == self._history[-5]
            and self._history[-3] == self._history[-6]
            and len({self._history[-1], self._history[-2], self._history[-3]}) == 3
        ):
            return True
        return False

    def _breakout_wall_kick(
        self, act: dict[str, Any], mx: float, my: float,
        fw: float, fh: float, sign: int, attack: str,
        best_dribble: str, pradius: float, pspeed: float,
        obstacles: list, observation: dict[str, Any]
    ) -> dict[str, Any]:
        k = act.get("kick")
        if not k:
            return act
        kdir = k.get("direction", "")
        pw = k.get("power", 1)
        is_wk = (
            (mx >= fw - 10.0 and "RIGHT" in kdir) or
            (mx <= 10.0 and "LEFT" in kdir) or
            (my <= 10.0 and "DOWN" in kdir) or
            (my >= fh - 10.0 and "UP" in kdir)
        )
        if is_wk:
            cur_wk = (kdir, pw)
            if cur_wk == self._last_wall_kick:
                self._wall_kick_repeats += 1
            else:
                self._last_wall_kick = cur_wk
                self._wall_kick_repeats = 1
        else:
            self._last_wall_kick = None
            self._wall_kick_repeats = 0

        if self._exact_hold_steps >= 8 and is_wk and self._wall_kick_repeats >= 2:
            my_id = observation["player_id"]
            open_h = "LEFT" if mx >= fw / 2.0 else "RIGHT"
            open_v = attack
            cand_kdirs = [f"{open_v}_{open_h}", open_v, open_h]
            cand_moves = [open_v, f"{open_v}_{open_h}", open_h, "STAY", best_dribble]
            valid_moves = [
                m for m in cand_moves
                if _is_move_safe(mx, my, m, fw, fh, pradius, pspeed, obstacles)
            ]
            if not valid_moves:
                valid_moves = ["STAY"]

            if self._fast_sim is not None:
                fast_st = self._fast_sim.from_env_observation(
                    observation, loose_ball_best_dist=self._loose_ball_best_dist
                )
                init_opp_score = fast_st.score_2 if my_id == "player_1" else fast_st.score_1
                best_action = None
                best_score = -999999.0

                for pw_cand in (3, 2):
                    for kd in cand_kdirs:
                        for mv in valid_moves:
                            cand_act = {"move": mv, "kick": {"direction": kd, "power": pw_cand}}
                            gives_opp_goal = False
                            min_ball_prog = 999999.0

                            for opp_m in MOVES:
                                sim_st = fast_st.clone()
                                p1_m = mv if my_id == "player_1" else opp_m
                                p2_m = opp_m if my_id == "player_1" else mv
                                p1_k = (kd, pw_cand) if my_id == "player_1" else None
                                p2_k = None if my_id == "player_1" else (kd, pw_cand)

                                self._fast_sim.step(sim_st, p1_m, p2_m, p1_k, p2_k)
                                opp_sc = sim_st.score_2 if my_id == "player_1" else sim_st.score_1
                                if opp_sc > init_opp_score:
                                    gives_opp_goal = True
                                    break

                                if sign > 0 and sim_st.ball_y < 12.0 and abs(sim_st.ball_x - 50.0) < 20.0:
                                    gives_opp_goal = True
                                    break
                                elif sign < 0 and sim_st.ball_y > fh - 12.0 and abs(sim_st.ball_x - 50.0) < 20.0:
                                    gives_opp_goal = True
                                    break

                                for _ in range(3):
                                    if sim_st.done:
                                        break
                                    o_dy = sim_st.ball_y - (sim_st.p2_y if my_id == "player_1" else sim_st.p1_y)
                                    opp_step_m = "UP" if o_dy > 0 else "DOWN"
                                    cand_step_m = "UP" if (sim_st.ball_y - (sim_st.p1_y if my_id == "player_1" else sim_st.p2_y)) * sign > 0 else "STAY"
                                    p1_step = cand_step_m if my_id == "player_1" else opp_step_m
                                    p2_step = opp_step_m if my_id == "player_1" else cand_step_m
                                    self._fast_sim.step(sim_st, p1_step, p2_step, None, None)
                                    opp_sc2 = sim_st.score_2 if my_id == "player_1" else sim_st.score_1
                                    if opp_sc2 > init_opp_score:
                                        gives_opp_goal = True
                                        break

                                if gives_opp_goal:
                                    break

                                prog = (sim_st.ball_y - my) * sign
                                if prog < min_ball_prog:
                                    min_ball_prog = prog

                            if not gives_opp_goal:
                                action_score = min_ball_prog * 10.0 + pw_cand * 5.0
                                if action_score > best_score:
                                    best_score = action_score
                                    best_action = cand_act

                if best_action is not None:
                    self._exact_hold_steps = 0
                    self._last_wall_kick = None
                    self._wall_kick_repeats = 0
                    return best_action

        return act

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Policy":
        params = {}
        if path:
            try:
                p = Path(path)
                if p.is_file():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    params = data.get("tuning_parameters", {})
            except Exception as err:
                _record_error(err)
        return cls(params)

    def choose_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        my_id = observation["player_id"]
        opp_id = observation["opponent_id"]
        st = observation["state"]
        field = st.get("field", {})
        fw = float(field.get("width", 100.0))
        fh = float(field.get("height", 140.0))
        gw = float(field.get("goal_width", 36.0))
        goal_left = (fw - gw) / 2
        goal_right = goal_left + gw
        pspeed = float(field.get("player_speed", 4.0))
        pradius = float(field.get("player_radius", 3.0))
        bradius = float(field.get("ball_radius", 1.5))
        bspeed = float(field.get("ball_speed", 8.0))
        kdists = (32.0, 64.0, 96.0)

        me = st["players"][my_id]
        opp = st["players"][opp_id]
        ball = st["ball"]
        attack = observation["attack_direction"]
        sign = 1 if attack == "UP" else -1

        my_goal_y = 0.0 if sign > 0 else fh
        opp_goal_y = fh if sign > 0 else 0.0

        mx, my = float(me["x"]), float(me["y"])
        ox, oy = float(opp["x"]), float(opp["y"])
        bx, by = float(ball["x"]), float(ball["y"])
        brem = float(ball.get("remaining_kick_distance", 0.0))

        possession = ball.get("possession")
        pstep = int(ball.get("possession_steps", 0))
        raw_obs = st.get("obstacles", [])
        obstacles = [Rectangle(o["x"], o["y"], o["width"], o["height"]) for o in raw_obs]

        opp_dist = math.hypot(ox - mx, oy - my)
        is_p1 = (my_id == "player_1")

        # Step 6 Item 3: Track loose-ball best distance
        if possession is not None or brem > 0.0:
            self._loose_ball_best_dist = None
        else:
            closest_d = min(math.hypot(mx - bx, my - by), math.hypot(ox - bx, oy - by))
            if self._loose_ball_best_dist is None or closest_d < self._loose_ball_best_dist - 0.25:
                self._loose_ball_best_dist = closest_d

        # ── Tactical State Machine ──
        iteration = int(st.get("iteration", 0))
        if iteration <= 1 or iteration < self._last_iter:
            self._sig.clear()
            self._stuck = 0
            self._last_best = None
            self._history.clear()
            self._exact_hold_pos = None
            self._exact_hold_steps = 0
            self._last_wall_kick = None
            self._wall_kick_repeats = 0
        self._last_iter = iteration
        if possession == my_id:
            if (self._exact_hold_pos is not None and
                abs(mx - self._exact_hold_pos[0]) < 0.2 and
                abs(my - self._exact_hold_pos[1]) < 0.2):
                self._exact_hold_steps += 1
            else:
                self._exact_hold_pos = (mx, my)
                self._exact_hold_steps = 1
        else:
            self._exact_hold_pos = None
            self._exact_hold_steps = 0
            self._last_wall_kick = None
            self._wall_kick_repeats = 0
        if possession == my_id and abs(mx - self._px) < 0.2 and abs(my - self._py) < 0.2:
            self._stuck += 1
        else:
            self._stuck = 0
        if possession == opp_id and abs(mx - self._px) < 0.2 and abs(my - self._py) < 0.2 and opp_dist < 9.0:
            self._dstuck += 1
        else:
            self._dstuck = 0
        self._px, self._py = mx, my
        self._history.append((round(mx, 1), round(my, 1)))
        if len(self._history) > 12:
            self._history.pop(0)
        score = st.get("score", {})
        my_score = int(score.get(my_id, 0))
        opp_score = int(score.get(opp_id, 0))
        lead = my_score - opp_score

        max_iters = int(st.get("maximum_iterations", 400))
        t_losing = int(max_iters * 0.75)
        t_winning = int(max_iters * 0.80)
        t_tied = int(max_iters * 0.85)

        if self._fast_sim is None or raw_obs != self._last_obs_obstacles:
            fast_obstacles = [FastObstacle(o["x"], o["y"], o["width"], o["height"]) for o in raw_obs]
            fast_cfg = FastConfig(
                field_width=fw, field_height=fh, goal_width=gw,
                player_radius=pradius, player_speed=pspeed,
                ball_radius=bradius, ball_speed=bspeed,
                maximum_iterations=max_iters
            )
            self._fast_sim = FastSimulator(fast_cfg, fast_obstacles)
            self._def_searcher = DefensiveSearch(self._fast_sim)
            self._shot_planner = ShotPlanner(self._fast_sim)
            self._last_obs_obstacles = raw_obs

        if iteration >= t_winning and lead > 0:
            mode = "WINNING_LATE"
            max_shot_dist = 36.0
            risky_shot_cutoff = 0.8
            min_pass_lead = 2.0
        elif iteration >= t_losing and lead < 0:
            mode = "LOSING_LATE"
            max_shot_dist = 52.0
            risky_shot_cutoff = -3.5
            min_pass_lead = 1.0
        elif iteration >= t_tied and lead == 0:
            mode = "TIED_LATE"
            max_shot_dist = 48.0
            risky_shot_cutoff = -2.5
            min_pass_lead = 1.2
        else:
            mode = "NORMAL"
            max_shot_dist = float(self.params.get("max_direct_shot_distance", 44.0))
            risky_shot_cutoff = float(self.params.get("risky_shot_margin_cutoff", -2.0))
            min_pass_lead = float(self.params.get("min_pass_lead_steps", 1.5))

        # ── 1. WITH BALL POSSESSION ────────────────────────────────────────
        if possession == my_id:
            guaranteed_kicks: list[tuple[int, int, dict[str, Any]]] = []
            candidate_scoring_kicks: list[tuple[float, int, int, dict[str, Any]]] = []
            candidate_self_passes: list[tuple[float, dict[str, Any]]] = []

            safe_moves = [m for m in MOVES if _is_move_safe(mx, my, m, fw, fh, pradius, pspeed, obstacles)]
            if not safe_moves:
                safe_moves = ["STAY"]

            def evaluate_action(mv_name: str, kdir: str, power: int) -> bool:
                """
                Evaluates action (mv_name, kdir, power) against all 9 opponent responses.
                Returns True iff a guaranteed scoring goal is confirmed.
                """
                mv = DIRECTIONS[mv_name]
                mux, muy = _unit(mv)
                p_new_x = mx + mux * pspeed
                p_new_y = my + muy * pspeed

                dv = DIRECTIONS[kdir]
                kux, kuy = _unit(dv)
                clearance = pradius + bradius + 0.05
                b_start_x = p_new_x + kux * clearance
                b_start_y = p_new_y + kuy * clearance

                b_vx = kux * bspeed
                b_vy = kuy * bspeed
                rem_d = kdists[power - 1]

                scored, conceded, trajectory = simulate_trajectory(
                    b_start_x, b_start_y, b_vx, b_vy, rem_d,
                    fw, fh, goal_left, goal_right, bradius, bspeed,
                    sign, obstacles, max_steps=15
                )

                if conceded or not trajectory:
                    return False

                act = {"move": mv_name, "kick": {"direction": kdir, "power": power}}

                # ── 1.5-Ply Adversarial Minimax over all 9 Opponent Responses ──
                worst_opp_margin = 999.0
                any_opp_intercepts = False
                worst_lead_margin = 999.0

                for om in MOVES:
                    ovx, ovy = DIRECTIONS[om]
                    oux, ouy = _unit((ovx, ovy))
                    if _is_move_safe(ox, oy, om, fw, fh, pradius, pspeed, obstacles):
                        nox = ox + oux * pspeed
                        noy = oy + ouy * pspeed
                    else:
                        nox, noy = ox, oy

                    # Step 1 Interception check:
                    step1_intercept = False
                    x1, y1, _ = trajectory[0]

                    # Temporal check: ball travel vs opponent travel to ray
                    if _ray_hits_circle(p_new_x, p_new_y, kux, kuy, 11.5, nox, noy, pradius + bradius + 0.15):
                        proj = (nox - p_new_x) * kux + (noy - p_new_y) * kuy
                        t_ball = max(0.05, proj / bspeed)
                        t_opp = math.hypot(nox - ox, noy - oy) / pspeed
                        if t_opp <= t_ball + 0.2:
                            step1_intercept = True

                    if math.hypot(nox - x1, noy - y1) <= pradius + bradius:
                        step1_intercept = True

                    om_min_margin = -999.0 if step1_intercept else 999.0
                    if step1_intercept:
                        any_opp_intercepts = True
                        worst_opp_margin = min(worst_opp_margin, -999.0)
                        continue

                    # Subsequent steps t >= 2
                    for bx_t, by_t, step_i in trajectory:
                        dist_opp_to_pt = math.hypot(nox - bx_t, noy - by_t)
                        reach_dist = (step_i - 1) * pspeed + pradius + bradius
                        margin = dist_opp_to_pt - reach_dist
                        if margin < om_min_margin:
                            om_min_margin = margin
                        if margin <= 0.0:
                            any_opp_intercepts = True
                            break

                    if om_min_margin < worst_opp_margin:
                        worst_opp_margin = om_min_margin

                    # Self-pass arrival lead for this opponent move
                    land_x, land_y, _ = trajectory[-1]
                    dist_opp_to_land = math.hypot(nox - land_x, noy - land_y)
                    t_opp_land = 1.0 + dist_opp_to_land / pspeed
                    dist_us_to_land = math.hypot(p_new_x - land_x, p_new_y - land_y)
                    t_us_land = 1.0 + dist_us_to_land / pspeed
                    lead_m = t_opp_land - t_us_land
                    if lead_m < worst_lead_margin:
                        worst_lead_margin = lead_m

                score_steps = len(trajectory)
                if scored and not conceded:
                    clear_margin = self.params.get("goal_clearance_margin", 0.5)
                    if not any_opp_intercepts and worst_opp_margin > clear_margin:
                        guaranteed_kicks.append((score_steps, power, act))
                        return True
                    else:
                        candidate_scoring_kicks.append((worst_opp_margin, score_steps, power, act))
                elif not conceded and not any_opp_intercepts and trajectory:
                    land_x, land_y, _ = trajectory[-1]
                    if 6.0 <= land_x <= fw - 6.0 and 6.0 <= land_y <= fh - 6.0:
                        dist_us_to_land = math.hypot(p_new_x - land_x, p_new_y - land_y)
                        run_blocked = False
                        inv_d = 1.0 / max(0.1, dist_us_to_land)
                        rux = (land_x - p_new_x) * inv_d
                        ruy = (land_y - p_new_y) * inv_d
                        for obs in obstacles:
                            cx = obs.x + obs.width / 2
                            cy = obs.y + obs.height / 2
                            obs_r = max(obs.width, obs.height) * 0.5 + pradius
                            if _ray_hits_circle(p_new_x, p_new_y, rux, ruy, dist_us_to_land, cx, cy, obs_r):
                                run_blocked = True
                                break
                        adj_worst_lead = worst_lead_margin - (1.25 if run_blocked else 0.0)
                        y_progress = sign * (land_y - my)

                        if adj_worst_lead >= min_pass_lead and y_progress >= 4.0:
                            pass_score = y_progress * 2.2 + adj_worst_lead * 3.0 - abs(land_x - 50.0) * 0.15
                            candidate_self_passes.append((pass_score, act))
                return False

            # Phase 1: Fast-path evaluation of aligned kicks (24 combinations)
            for kdir in KICK_DIRS:
                pref_move = kdir if kdir in MOVES else attack
                m_aligned = _best_safe_move(mx, my, pref_move, fw, fh, pradius, pspeed, obstacles)
                for power in (1, 2, 3):
                    evaluate_action(m_aligned, kdir, power)

            # Phase 2: Decoupled 216-Action Search (evaluates sidestep curling shots if no direct instant goal)
            if not guaranteed_kicks:
                for mv_name in safe_moves:
                    for kdir in KICK_DIRS:
                        if mv_name == kdir:
                            continue
                        for power in (1, 2, 3):
                            evaluate_action(mv_name, kdir, power)

            # Priority 1: Guaranteed scoring kick
            if guaranteed_kicks:
                guaranteed_kicks.sort(key=lambda item: (item[0], item[1]))
                return guaranteed_kicks[0][2]

            # Phase 3 / Section 4.4: Exact-physics ShotPlanner guaranteed shot
            if self.params.get("flag_shot_planner", True) and self._shot_planner:
                try:
                    fast_st = self._fast_sim.from_env_observation(
                        observation, loose_ball_best_dist=self._loose_ball_best_dist
                    )
                    planned_shot = self._shot_planner.find_guaranteed_shot(
                        fast_st, my_id, opp_id, opp_goal_y
                    )
                    if planned_shot is not None:
                        return planned_shot
                except Exception as err:
                    _record_error(err)

            # Priority 2: Candidate scoring kick when in shooting range
            dist_to_opp_goal = fh - my if sign > 0 else my
            if dist_to_opp_goal <= max_shot_dist and candidate_scoring_kicks:
                candidate_scoring_kicks.sort(key=lambda item: (-item[0], item[1]))
                best_cand = None
                for _margin, _st, _pw, _act in candidate_scoring_kicks:
                    if self._sig_n(_act, mx, my) < 2:
                        best_cand = (_margin, _st, _pw, _act)
                        break
                if best_cand is not None and best_cand[0] > risky_shot_cutoff:
                    return self._sig_mark(best_cand[3], mx, my)

            # Priority 3: Dribble path evaluation
            fwd_moves = [attack, f"{attack}_LEFT", f"{attack}_RIGHT", "LEFT", "RIGHT"]
            safe_fwd = [m for m in fwd_moves if _is_move_safe(mx, my, m, fw, fh, pradius, pspeed, obstacles)]
            opp_ahead = (sign * (oy - my) > 0 and abs(ox - mx) < 14.0 and opp_dist < 35.0)

            best_dribble = "STAY"
            best_dribble_score = -99999.0
            for m in (safe_fwd if safe_fwd else MOVES):
                if not _is_move_safe(mx, my, m, fw, fh, pradius, pspeed, obstacles):
                    continue
                uv = DIRECTIONS[m]
                nmx = mx + uv[0] * pspeed
                nmy = my + uv[1] * pspeed
                n_dist_goal = fh - nmy if sign > 0 else nmy
                n_dist_opp = math.hypot(ox - nmx, oy - nmy)
                s = -n_dist_goal * 2.0
                if opp_ahead:
                    s += min(20.0, n_dist_opp) * 1.5
                if (self._stuck >= 2 or self._is_in_periodic_cycle()) and m == self._last_best:
                    s -= 2000.0
                if s > best_dribble_score:
                    best_dribble_score = s
                    best_dribble = m
            self._last_best = best_dribble

            # Priority 4: Safe Open-Field Carry / Evasion Dribble
            can_carry = (pstep < 2 or opp_dist > self.params.get("safe_dribble_opp_dist", 7.0)) and pstep < 8
            if can_carry and not opp_ahead and best_dribble != "STAY":
                return {"move": best_dribble}

            # Priority 5: 2-Ply Kick-and-Chase (Self-Pass) when obstructed or pressured
            if candidate_self_passes:
                candidate_self_passes.sort(key=lambda item: -item[0])
                best_pass_score, best_pass_act = candidate_self_passes[0]
                for _sc, _act in candidate_self_passes:
                    if self._sig_n(_act, mx, my) < 2:
                        best_pass_score, best_pass_act = _sc, _act
                        break
                else:
                    best_pass_act = None
                if best_pass_act is not None and (opp_ahead or opp_dist <= 10.0 or pstep >= 7):
                    return self._sig_mark(best_pass_act, mx, my)

            # Carry with evasion if possible
            if can_carry and best_dribble != "STAY":
                return {"move": best_dribble}

            # Priority 6: Pressure Clearance
            best_clear = None
            best_clear_score = -99999.0

            for kdir in KICK_DIRS:
                dv = DIRECTIONS[kdir]
                ku = _unit(dv)
                dx_op = ox - mx
                dy_op = oy - my
                s_proj = dx_op * ku[0] + dy_op * ku[1]
                perp_d = abs(dx_op * ku[1] - dy_op * ku[0])
                if s_proj > 0 and s_proj < 24.0 and perp_d < (pradius + bradius + pspeed):
                    continue

                for power in (1, 2, 3):
                    t_bx = mx + ku[0] * kdists[power - 1]
                    t_by = my + ku[1] * kdists[power - 1]
                    if t_bx < 6.0 or t_bx > fw - 6.0 or t_by < 6.0 or t_by > fh - 6.0:
                        continue

                    dist_opp_land = math.hypot(ox - t_bx, oy - t_by)
                    dist_us_land = math.hypot(mx - t_bx, my - t_by)

                    if dist_opp_land < dist_us_land + 4.0:
                        continue

                    if abs(t_bx - 50.0) < 22.0 and (t_by < 18.0 if sign > 0 else t_by > fh - 18.0):
                        continue

                    sc = (dist_opp_land - dist_us_land) * 2.0 + abs(t_bx - 50.0) * 0.3
                    if sign > 0 and t_by > my:
                        sc += 5.0
                    elif sign < 0 and t_by < my:
                        sc += 5.0

                    sc -= 1000.0 * min(1, self._sig_n({"kick": {"direction": kdir, "power": power}}, mx, my) // 2)

                    if sc > best_clear_score:
                        best_clear_score = sc
                        best_clear = {"move": best_dribble, "kick": {"direction": kdir, "power": power}}

            if best_clear:
                return self._breakout_wall_kick(
                    self._sig_mark(best_clear, mx, my),
                    mx, my, fw, fh, sign, attack, best_dribble, pradius, pspeed, obstacles, observation
                )

            perp_dir = "RIGHT" if ox <= mx else "LEFT"
            if (mx <= 5.0 and perp_dir == "LEFT") or (mx >= fw - 5.0 and perp_dir == "RIGHT"):
                in_defensive_box = (my < 25.0 if sign > 0 else my > fh - 25.0)
                if not in_defensive_box:
                    return {"move": best_dribble}
            return self._breakout_wall_kick(
                {"move": best_dribble, "kick": {"direction": perp_dir, "power": 1}},
                mx, my, fw, fh, sign, attack, best_dribble, pradius, pspeed, obstacles, observation
            )

        # ── 2. WITHOUT BALL POSSESSION ─────────────────────────────────────
        if possession == opp_id:
            caught_upfield = (sign > 0 and my >= oy) or (sign < 0 and my <= oy)
            stuck_def = self._dstuck >= 10
            if caught_upfield and not stuck_def:
                ret_dir = "DOWN" if sign > 0 else "UP"
                return {"move": _best_safe_move(mx, my, ret_dir, fw, fh, pradius, pspeed, obstacles)}

            vx = 50.0 - ox
            vy = my_goal_y - oy
            ux, uy = _unit((vx, vy))
            dist_opp_goal = math.hypot(vx, vy)

            press_thresh = self.params.get("midfield_press_threshold", 42.0)
            cush_max = self.params.get("midfield_press_cushion_max", 10.0)
            cush_min = self.params.get("midfield_press_cushion_min", 7.0)

            if self._dstuck >= 10:
                _bx = ox + ux * 16.0
                _by = oy + uy * 16.0
                if self._dstuck >= 18:
                    self._dstuck = 0
                return {"move": _best_safe_move(mx, my, _direction_toward(_bx - mx, _by - my), fw, fh, pradius, pspeed, obstacles)}

            # Tactical defending behavior based on state machine
            if mode in ("LOSING_LATE", "TIED_LATE") and pstep >= 2:
                # Early aggressive pressing to force tackle turnovers!
                target_x = ox + ux * 2.5
                target_y = oy + uy * 2.5
            elif pstep >= 8:
                f_ux, f_uy = (0.0, -1.0 if sign > 0 else 1.0)
                target_x = ox + f_ux * 6.0
                target_y = oy + f_uy * 6.0
            elif pstep >= 3 and (dist_opp_goal <= 50.0 or mode in ("LOSING_LATE", "TIED_LATE")):
                target_x = ox + ux * 3.2
                target_y = oy + uy * 3.2
            elif dist_opp_goal <= press_thresh:
                cushion = min(cush_max, max(cush_min, dist_opp_goal * 0.22))
                target_x = ox + ux * cushion
                target_y = oy + uy * cushion
            else:
                anchor_ratio = 0.35
                target_x = ox * anchor_ratio + 50.0 * (1.0 - anchor_ratio)
                target_y = my_goal_y + sign * 22.0

            def_move = _best_safe_move(
                mx, my, _direction_toward(target_x - mx, target_y - my),
                fw, fh, pradius, pspeed, obstacles
            )

            # Phase 3: Defensive joint-action minimax search with failsafe heuristic fallback
            if self.params.get("flag_defensive_search", True) and self._def_searcher:
                try:
                    fast_st = self._fast_sim.from_env_observation(
                        observation, loose_ball_best_dist=self._loose_ball_best_dist
                    )
                    searched_move = self._def_searcher.search_best_defensive_move(
                        fast_st, my_id, opp_id, my_goal_y,
                        fallback_move=def_move, time_limit_ms=25.0
                    )
                    if searched_move in MOVES and searched_move != "STAY":
                        def_move = searched_move
                except Exception as err:
                    _record_error(err)

            def_act: dict[str, Any] = {"move": def_move}

            if self.params.get("flag_tackle_plus_kick", True):
                if opp_dist <= (pradius * 2 + 0.15) and pstep >= 3:
                    counter_kdir = _direction_toward(50.0 - mx, opp_goal_y - my)
                    if counter_kdir == "STAY":
                        counter_kdir = attack
                    def_act["kick"] = {"direction": counter_kdir, "power": 3}

            return def_act

        # ── 3. LOOSE BALL INTERCEPTION ────────────────────────────────────
        loose_steps = int(ball.get("loose_ball_steps", 0))
        dist_me_ball = math.hypot(mx - bx, my - by)
        dist_opp_ball = math.hypot(ox - bx, oy - by)

        # Late-Game Defensive Shape Management:
        # When protecting a lead late in the match and the loose ball is deep in the opponent half
        # while the opponent is distant, maintain disciplined goal-side defensive positioning.
        if mode == "WINNING_LATE" and (sign * (by - fh / 2.0) > 10.0) and dist_opp_ball > 20.0:
            tx = 50.0
            ty = my_goal_y + sign * 25.0
            return {
                "move": _best_safe_move(
                    mx, my, _direction_toward(tx - mx, ty - my),
                    fw, fh, pradius, pspeed, obstacles
                )
            }

        # Deterministic Player-Priority Arrival Modeling:
        # Accounts for the engine's resolution order favoring Player 1 on exact distance/contact ties.
        t_me = dist_me_ball / pspeed
        t_opp = dist_opp_ball / pspeed

        should_retreat = False
        if is_p1:
            # Player 1 wins ties; contest 50/50 loose balls, retreat only if opponent is ahead by > 0.6 steps
            if t_opp + 0.6 < t_me:
                should_retreat = True
        else:
            # Player 2 loose-ball rule: contest unless opponent is ahead by > 0.8 steps
            if t_opp + 0.8 < t_me:
                should_retreat = True

        if should_retreat and ((sign > 0 and my > my_goal_y + 15.0) or (sign < 0 and my < my_goal_y - 15.0)):
            vx = 50.0 - bx
            vy = my_goal_y - by
            ux, uy = _unit((vx, vy))
            cush = min(20.0, max(8.0, math.hypot(vx, vy) * 0.3))
            tx = bx + ux * cush
            ty = by + uy * cush
        elif loose_steps >= 18 and (ball.get("status") == "stationary" or brem <= 0):
            tx = fw / 2.0
            ty = fh / 2.0
        elif ball.get("status") == "moving" or brem > 0:
            b_vx = float(ball.get("velocity", {}).get("x", 0.0))
            b_vy = float(ball.get("velocity", {}).get("y", 0.0))
            _, _, traj = simulate_trajectory(
                bx, by, b_vx, b_vy, brem,
                fw, fh, goal_left, goal_right, bradius, bspeed,
                sign, obstacles, max_steps=15
            )
            intercept_pt = (bx, by)
            for cur_bx, cur_by, step_i in traj:
                dist_to_ball_pt = math.hypot(mx - cur_bx, my - cur_by)
                if dist_to_ball_pt <= step_i * pspeed + pradius + bradius:
                    intercept_pt = (cur_bx, cur_by)
                    break
                intercept_pt = (cur_bx, cur_by)

            tx, ty = intercept_pt
        else:
            tx, ty = bx, by

        return {
            "move": _best_safe_move(
                mx, my, _direction_toward(tx - mx, ty - my),
                fw, fh, pradius, pspeed, obstacles
            )
        }
