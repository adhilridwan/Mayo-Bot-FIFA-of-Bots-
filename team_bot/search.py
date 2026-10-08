
from __future__ import annotations

import math
import time
from typing import Any

try:
    from .simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState
except (ImportError, ValueError):
    try:
        from my_team.team_bot.simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState
    except ImportError:
        from simulator import DIRECTIONS, FastConfig, FastObstacle, FastSimulator, FastState

MOVES = list(DIRECTIONS)
PLAYERS = ("player_1", "player_2")


def _unit(v: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(v[0], v[1])
    return (0.0, 0.0) if length == 0.0 else (v[0] / length, v[1] / length)


def evaluate_defensive_leaf(
    st: FastState,
    my_id: str,
    opp_id: str,
    my_goal_y: float,
    fw: float = 100.0,
    fh: float = 140.0
) -> float:

    my_score = st.score_1 if my_id == "player_1" else st.score_2
    opp_score = st.score_2 if my_id == "player_1" else st.score_1

    # Terminal or goal scored
    if opp_score > 0:
        return -10000.0
    if my_score > 0:
        return 10000.0

    # Possession value
    if st.possession == my_id:
        return 1500.0
    elif st.possession == opp_id:
        val = -200.0
    else:
        val = 0.0

    mx = st.p1_x if my_id == "player_1" else st.p2_x
    my = st.p1_y if my_id == "player_1" else st.p2_y
    ox = st.p2_x if my_id == "player_1" else st.p1_x
    oy = st.p2_y if my_id == "player_1" else st.p1_y
    bx = st.ball_x
    by = st.ball_y

    sign = 1 if my_goal_y == 0.0 else -1

    # Distance of opponent to our goal center
    dist_opp_goal = math.hypot(ox - 50.0, oy - my_goal_y)
    dist_me_opp = math.hypot(mx - ox, my - oy)
    dist_me_ball = math.hypot(mx - bx, my - by)

    val += dist_opp_goal * 3.0
    val -= dist_me_opp * 1.5

    # Goalkeeping geometry: position between opponent/ball and goal mouth
    is_behind_opp = (my < oy) if sign > 0 else (my > oy)
    if is_behind_opp:
        val += 80.0
        # Minimize perpendicular deviation from shooting line (ox, oy) -> (50, my_goal_y)
        dx_g = 50.0 - ox
        dy_g = my_goal_y - oy
        g_len = math.hypot(dx_g, dy_g)
        if g_len > 1e-4:
            # Perpendicular distance of defender to shot line
            perp_dist = abs((50.0 - ox) * (oy - my) - (ox - mx) * (my_goal_y - oy)) / g_len
            val -= perp_dist * 2.5
    else:
        # Heavily penalize being caught upfield
        val -= 300.0

    # Deny surface access (Section 4.3): penalize opponent being close to walls with ball
    if st.possession == opp_id:
        if ox <= 12.0 or ox >= fw - 12.0:
            val -= 60.0

    # Loose ball urgency
    if st.possession is None:
        val -= dist_me_ball * 2.0

    return val


class DefensiveSearch:
    def __init__(self, sim: FastSimulator):
        self.sim = sim
        self.fw = sim.fw
        self.fh = sim.fh
        self.gw = sim.gw
        self.pr = sim.pr
        self.ps = sim.ps
        self.br = sim.br
        self.bs = sim.bs

    def search_best_defensive_move(
        self,
        current_state: FastState,
        my_id: str,
        opp_id: str,
        my_goal_y: float,
        fallback_move: str = "STAY",
        time_limit_ms: float = 25.0
    ) -> str:
        t0 = time.perf_counter()
        best_move = fallback_move
        best_minimax_val = -999999.0

        mx = current_state.p1_x if my_id == "player_1" else current_state.p2_x
        my = current_state.p1_y if my_id == "player_1" else current_state.p2_y
        ox = current_state.p2_x if my_id == "player_1" else current_state.p1_x
        oy = current_state.p2_y if my_id == "player_1" else current_state.p1_y

        # Candidate moves for defender
        my_legal = [
            m for m in MOVES
            if self.sim.is_valid_player_pos(
                mx + DIRECTIONS[m][0] * self.ps,
                my + DIRECTIONS[m][1] * self.ps
            )
        ]
        if not my_legal:
            return fallback_move

        # Move ordering for defender: prioritize fallback_move and moves toward goal-side anchor
        sign = 1 if my_goal_y == 0.0 else -1
        anchor_x = ox * 0.4 + 50.0 * 0.6
        anchor_y = my_goal_y + sign * 18.0

        def my_move_priority(m: str) -> float:
            if m == fallback_move:
                return 1000.0
            vx, vy = DIRECTIONS[m]
            nmx = mx + vx * self.ps
            nmy = my + vy * self.ps
            return -math.hypot(nmx - anchor_x, nmy - anchor_y)

        my_legal.sort(key=my_move_priority, reverse=True)

        # Generate candidate opponent threat actions
        opp_threat_actions: list[dict[str, Any]] = []

        opp_legal_moves = [
            m for m in MOVES
            if self.sim.is_valid_player_pos(
                ox + DIRECTIONS[m][0] * self.ps,
                oy + DIRECTIONS[m][1] * self.ps
            )
        ]
        if not opp_legal_moves:
            opp_legal_moves = ["STAY"]

        # Opponent move-only threats (advances towards goal, towards walls, or pressing)
        goal_dir = "DOWN" if my_goal_y == 0.0 else "UP"
        for om in opp_legal_moves:
            opp_threat_actions.append({"move": om})

        # If opponent has ball, add dangerous kicks (direct shots on our goal)
        if current_state.possession == opp_id:
            shoot_dirs = (
                ["DOWN", "DOWN_LEFT", "DOWN_RIGHT", "LEFT", "RIGHT"]
                if my_goal_y == 0.0
                else ["UP", "UP_LEFT", "UP_RIGHT", "LEFT", "RIGHT"]
            )
            for kdir in shoot_dirs:
                for pwr in (1, 2, 3):
                    # Direct kick from best move
                    opp_threat_actions.append({
                        "move": goal_dir if goal_dir in opp_legal_moves else "STAY",
                        "kick": {"direction": kdir, "power": pwr}
                    })

        # Evaluate minimax: max_my (min_opp (utility))
        for my_m in my_legal:
            worst_opp_val = 999999.0

            for opp_act in opp_threat_actions:
                st1 = current_state.clone()
                opp_m = opp_act["move"]
                opp_k = (opp_act["kick"]["direction"], opp_act["kick"]["power"]) if "kick" in opp_act else None

                if my_id == "player_1":
                    p1_m, p1_k = my_m, None
                    p2_m, p2_k = opp_m, opp_k
                else:
                    p1_m, p1_k = opp_m, opp_k
                    p2_m, p2_k = my_m, None

                self.sim.step(st1, p1_m, p2_m, p1_k, p2_k)

                leaf_v = evaluate_defensive_leaf(st1, my_id, opp_id, my_goal_y, self.fw, self.fh)
                if leaf_v < worst_opp_val:
                    worst_opp_val = leaf_v

                # Time guard check inside inner loop
                if (time.perf_counter() - t0) * 1000.0 > time_limit_ms:
                    break

            if worst_opp_val > best_minimax_val:
                best_minimax_val = worst_opp_val
                best_move = my_m

            # Time guard check inside outer loop
            if (time.perf_counter() - t0) * 1000.0 > time_limit_ms:
                break

        return best_move
