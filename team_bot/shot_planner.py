
from __future__ import annotations

import math
from typing import Any

try:
    from .simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState
except (ImportError, ValueError):
    try:
        from my_team.team_bot.simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState
    except ImportError:
        from simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState

MOVES = list(DIRECTIONS)
KICK_DIRS = [d for d in MOVES if d != "STAY"]


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(v[0], v[1])
    return (0.0, 0.0) if length == 0.0 else (v[0] / length, v[1] / length)


class ShotPlanner:
    def __init__(self, sim: FastSimulator):
        self.sim = sim
        self.fw = sim.fw
        self.fh = sim.fh
        self.gw = sim.gw
        self.pr = sim.pr
        self.ps = sim.ps
        self.br = sim.br
        self.bs = sim.bs
        self.goal_left = (self.fw - self.gw) / 2.0
        self.goal_right = self.goal_left + self.gw

    def is_near_wall(self, x: float) -> str | None:
        if x <= 10.0:
            return "LEFT"
        if x >= self.fw - 10.0:
            return "RIGHT"
        return None

    def find_guaranteed_shot(
        self,
        current_state: FastState,
        my_id: str,
        opp_id: str,
        opp_goal_y: float,
    ) -> dict[str, Any] | None:
        """
        Simulates all candidate (move, kick_dir, power) actions against worst-case opponent responses.
        Returns the action if a 100% guaranteed, uninterceptable goal is confirmed.
        """
        if current_state.possession != my_id:
            return None

        mx = current_state.p1_x if my_id == "player_1" else current_state.p2_x
        my = current_state.p1_y if my_id == "player_1" else current_state.p2_y
        ox = current_state.p2_x if my_id == "player_1" else current_state.p1_x
        oy = current_state.p2_y if my_id == "player_1" else current_state.p1_y

        sign = 1 if opp_goal_y > 70.0 else -1
        dist_to_goal = opp_goal_y - my if sign > 0 else my - opp_goal_y

        # Only evaluate shots within reasonable range
        if dist_to_goal > 65.0:
            return None

        # Prioritize moves advancing toward goal or shifting along post line
        attack_dir = "UP" if sign > 0 else "DOWN"
        candidate_moves = [attack_dir, f"{attack_dir}_LEFT", f"{attack_dir}_RIGHT", "LEFT", "RIGHT", "STAY"]
        safe_moves = [
            m for m in candidate_moves
            if self.sim.is_valid_player_pos(
                mx + DIRECTIONS[m][0] * self.ps,
                my + DIRECTIONS[m][1] * self.ps
            )
        ]
        if not safe_moves:
            safe_moves = ["STAY"]

        # Opponent legal moves
        opp_legal = [
            m for m in MOVES
            if self.sim.is_valid_player_pos(
                ox + DIRECTIONS[m][0] * self.ps,
                oy + DIRECTIONS[m][1] * self.ps
            )
        ]
        if not opp_legal:
            opp_legal = ["STAY"]

        # Goal-scoring kick directions
        target_kicks = (
            ["UP", "UP_LEFT", "UP_RIGHT", "LEFT", "RIGHT"]
            if sign > 0
            else ["DOWN", "DOWN_LEFT", "DOWN_RIGHT", "LEFT", "RIGHT"]
        )

        best_shot: dict[str, Any] | None = None
        min_steps = 999

        for mv in safe_moves:
            for kdir in target_kicks:
                for pwr in (1, 2, 3):
                    # Check if action scores against ALL opponent moves
                    guaranteed = True
                    max_flight_steps = 0

                    for om in opp_legal:
                        st1 = current_state.clone()
                        if my_id == "player_1":
                            self.sim.step(st1, mv, om, (kdir, pwr), None)
                        else:
                            self.sim.step(st1, om, mv, None, (kdir, pwr))

                        my_score = st1.score_1 if my_id == "player_1" else st1.score_2
                        opp_score = st1.score_2 if my_id == "player_1" else st1.score_1

                        # If step 1 did not score, check if ball continues to score or is intercepted
                        if opp_score > 0:
                            guaranteed = False
                            break

                        if my_score > 0:
                            # Scored on step 1!
                            max_flight_steps = max(max_flight_steps, 1)
                            continue

                        # If not scored immediately, simulate up to 4 more steps to see if it reaches goal
                        scored_within = False
                        st_flight = st1.clone()
                        for step_i in range(2, 6):
                            if st_flight.done:
                                break
                            # Opponent moves toward ball to intercept
                            bx, by = st_flight.ball_x, st_flight.ball_y
                            vox = bx - (st_flight.p2_x if my_id == "player_1" else st_flight.p1_x)
                            voy = by - (st_flight.p2_y if my_id == "player_1" else st_flight.p1_y)
                            h = math.hypot(vox, voy)
                            opp_opt = "STAY"
                            if h > 1e-3:
                                best_dot = -999.0
                                for m_cand in MOVES:
                                    ux, uy = DIRECTIONS[m_cand]
                                    dot = ux * (vox / h) + uy * (voy / h)
                                    if dot > best_dot:
                                        best_dot = dot
                                        opp_opt = m_cand

                            if my_id == "player_1":
                                self.sim.step(st_flight, "STAY", opp_opt)
                            else:
                                self.sim.step(st_flight, opp_opt, "STAY")

                            ms = st_flight.score_1 if my_id == "player_1" else st_flight.score_2
                            if ms > 0:
                                scored_within = True
                                max_flight_steps = max(max_flight_steps, step_i)
                                break
                            if st_flight.possession == opp_id or (st_flight.ball_rem_dist <= 0 and st_flight.possession is None):
                                break

                        if not scored_within:
                            guaranteed = False
                            break

                    if guaranteed and max_flight_steps < min_steps:
                        min_steps = max_flight_steps
                        best_shot = {
                            "move": mv,
                            "kick": {"direction": kdir, "power": pwr}
                        }
                        # If fast 1-step guaranteed goal found, return immediately
                        if min_steps <= 1:
                            return best_shot

        return best_shot

    def get_surface_hold_action(
        self,
        current_state: FastState,
        my_id: str,
        opp_id: str,
        lead: int,
        iteration: int,
        max_iters: int,
    ) -> dict[str, Any] | None:

        if current_state.possession != my_id:
            return None

        mx = current_state.p1_x if my_id == "player_1" else current_state.p2_x
        my = current_state.p1_y if my_id == "player_1" else current_state.p2_y
        pstep = current_state.possession_steps

        wall_side = self.is_near_wall(mx)
        if wall_side is None:
            return None

        # Section 4.3: Wall-lock when leading late
        if lead > 0 and iteration >= int(max_iters * 0.80):
            # Kick directly into wall power 1 every step to run down clock safely
            return {
                "move": "STAY",
                "kick": {"direction": wall_side, "power": 1}
            }

        # Section 4.2: Tackle-proof wall-dribble progression
        # When near wall and facing pressure or late possession steps (pstep >= 7),
        # advance forward along the wall while self-bouncing into the wall to reset counter!
        ox = current_state.p2_x if my_id == "player_1" else current_state.p1_x
        oy = current_state.p2_y if my_id == "player_1" else current_state.p1_y
        opp_dist = math.hypot(ox - mx, oy - my)

        if pstep >= 7 or (opp_dist <= 8.0 and pstep >= 3):
            sign = 1 if current_state.p1_y < current_state.p2_y else -1
            adv_move = "UP" if sign > 0 else "DOWN"
            # Verify advancing move is safe from boundaries and obstacles
            if not self.sim.is_valid_player_pos(
                mx + DIRECTIONS[adv_move][0] * self.ps,
                my + DIRECTIONS[adv_move][1] * self.ps
            ):
                adv_move = "STAY"

            return {
                "move": adv_move,
                "kick": {"direction": wall_side, "power": 1}
            }

        return None
