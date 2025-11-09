"""
Ultra-Heuristic Toroidal Tron Agent (18x20)
- No ML, pure logic
- Torus-aware everywhere
- Opening book for fixed spawns
- Space control (full BFS w/ cap)
- Opponent simulation + trap scoring
- Adaptive danger + boost usage
"""

import os
import random
from collections import deque
from typing import Tuple, List, Set, Dict

from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------
# CONSTANTS / CONFIG
# ---------------------------------------------------------------------
BOARD_WIDTH = 20
BOARD_HEIGHT = 18

PARTICIPANT = os.getenv("PARTICIPANT", "Chicken")
AGENT_NAME = os.getenv("AGENT_NAME", "UltraHeuristicAgent")

MAX_FLOOD = 240  # cap for BFS space calc (board is 360 cells, 240 is good enough)

# ---------------------------------------------------------------------
# GAME STATE
# ---------------------------------------------------------------------
game_state = {
    "board": {"width": BOARD_WIDTH, "height": BOARD_HEIGHT, "grid": None},
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "agent1_alive": True,
    "agent2_alive": True,
    "turn_count": 0,
    "player_number": 1,
}

# logical state we maintain across turns
strategy_state = {
    "space_advantage": 0.0,
    "danger_level": 0.0,
    "opponent_aggression": 0.5,
    "move_history": deque(maxlen=60),
    "boost_used_turns": [],
    "opening_book_active": True,
    "opening_key": None,
}


# ---------------------------------------------------------------------
# FLASK ENDPOINTS
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400

    # update core
    game_state.update(data)

    # normalize board
    if "board" in data and isinstance(data["board"], list):
        game_state["board"] = {
            "height": len(data["board"]),
            "width": len(data["board"][0]) if data["board"] else BOARD_WIDTH,
            "grid": data["board"],
        }

    # normalize trails to tuples
    if "agent1_trail" in data:
        game_state["agent1_trail"] = [tuple(p) for p in data["agent1_trail"]]
    if "agent2_trail" in data:
        game_state["agent2_trail"] = [tuple(p) for p in data["agent2_trail"]]

    # analyze current state for adaptive params
    try:
        analyze_game_state()
        detect_opening_if_any()
    except Exception as e:
        print("analysis error:", e)

    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    move = decide_move(player_number)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    # reset strategy for next game
    strategy_state.update({
        "space_advantage": 0.0,
        "danger_level": 0.0,
        "opponent_aggression": 0.5,
        "move_history": deque(maxlen=60),
        "boost_used_turns": [],
        "opening_book_active": True,
        "opening_key": None,
    })
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------
# DECISION LAYER
# ---------------------------------------------------------------------
def decide_move(player_number: int) -> str:
    my_trail, my_boosts, opp_trail = get_player_info(player_number)
    turn_count = game_state.get("turn_count", 0)

    if not my_trail:
        return "RIGHT"

    head = my_trail[-1]
    current_dir = get_current_direction(my_trail)
    board = game_state.get("board", {})
    h = board.get("height", BOARD_HEIGHT)
    w = board.get("width", BOARD_WIDTH)
    blocked = set(my_trail) | set(opp_trail)

    # only if opponent exists
    if opp_trail:
        opp_head = opp_trail[-1]
        if players_are_separated(head, opp_head, blocked, h, w, turn_count):
            move = efficient_path_move(head, current_dir, my_trail, h, w)
            strategy_state["move_history"].append(move)
            return add_boost_decision(move, my_boosts, turn_count)

    # opening, if any
    if strategy_state.get("opening_book_active", False):
        book_move = get_opening_book_move(head, current_dir, my_trail, opp_trail, turn_count)
        if book_move:
            strategy_state["move_history"].append(book_move)
            return add_boost_decision(book_move, my_boosts, turn_count)
        # --- optional: deliberate self-trap if dominant and safe ---
    if opp_trail:
        opp_head = opp_trail[-1]
        space_adv = strategy_state.get("space_advantage", 0.0)
        dist_to_opp = manhattan_torus(head, opp_head, h, w)

        # Lower threshold to make it more opportunistic
        if space_adv > 0.17 and dist_to_opp > 5:
            if should_self_trap(head, opp_head, my_trail, opp_trail, h, w):
                sealing_move = find_sealing_move(head, my_trail, opp_trail, h, w)
                if sealing_move:
                    print(f"🧱 Self-trap: Adv={space_adv:.2f} Dist={dist_to_opp}")
                    strategy_state["move_history"].append(sealing_move)
                    return add_boost_decision(sealing_move, my_boosts, turn_count)

    # normal heuristic
    move = choose_best_move(head, current_dir, my_trail, opp_trail, turn_count)
    strategy_state["move_history"].append(move)
    return add_boost_decision(move, my_boosts, turn_count)


def choose_best_move(head, current_dir, my_trail, opp_trail, turn_count) -> str:
    board = game_state.get("board", {})
    h = board.get("height", BOARD_HEIGHT)
    w = board.get("width", BOARD_WIDTH)

    safe_moves = get_safe_moves(head, current_dir, my_trail, opp_trail, h, w)
    if not safe_moves:
        # fall back to any non-180 move
        safe_moves = [d for d in ["UP", "DOWN", "LEFT", "RIGHT"]
                      if d != opposite_dir(current_dir)]
        if not safe_moves:
            return current_dir

    scored = {}
    for m in safe_moves:
        scored[m] = score_move(head, m, my_trail, opp_trail, h, w, turn_count)

    return max(scored.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------
# OPENING BOOK
# ---------------------------------------------------------------------
def detect_opening_if_any():
    """
    If both players always spawn at the same two cells, we can detect that on turn 0–1
    and set a key. This lets us hardcode a safe expansion for first few turns.
    """
    turn = game_state.get("turn_count", 0)
    if turn > 2:
        # too late to detect
        return

    a1 = game_state.get("agent1_trail", [])
    a2 = game_state.get("agent2_trail", [])
    if not a1 or not a2:
        return

    p1 = a1[0]
    p2 = a2[0]
    # key is sorted pair of spawns
    key = tuple(sorted([p1, p2]))
    strategy_state["opening_key"] = key


OFFICIAL_SPAWN_P1 = (1, 2)
OFFICIAL_SPAWN_P2 = (17, 15)

def get_opening_book_move(head, current_dir, my_trail, opp_trail, turn_count: int):
    # stop after a few moves
    if turn_count > 6:
        strategy_state["opening_book_active"] = False
        return None

    if not my_trail or not opp_trail:
        return None

    my_start = my_trail[0]
    opp_start = opp_trail[0]

    # simple safe-ish sequences (tweak these for your map)
    p1_seq = ["RIGHT", "RIGHT", "UP", "UP", "RIGHT", "RIGHT", "DOWN"]
    p2_seq = ["LEFT", "LEFT", "DOWN", "DOWN", "LEFT", "LEFT", "UP"]

    head_now = my_trail[-1]

    if my_start == OFFICIAL_SPAWN_P1 and opp_start == OFFICIAL_SPAWN_P2:
        if turn_count < len(p1_seq):
            m = p1_seq[turn_count]
            if is_move_safe(head_now, m, my_trail, opp_trail):
                return m
            else:
                strategy_state["opening_book_active"] = False
                return None

    elif my_start == OFFICIAL_SPAWN_P2 and opp_start == OFFICIAL_SPAWN_P1:
        if turn_count < len(p2_seq):
            m = p2_seq[turn_count]
            if is_move_safe(head_now, m, my_trail, opp_trail):
                return m
            else:
                strategy_state["opening_book_active"] = False
                return None
    else:
        strategy_state["opening_book_active"] = False
        return None



# ---------------------------------------------------------------------
# MOVE SCORING
# ---------------------------------------------------------------------
def score_move(head, direction, my_trail, opp_trail, h, w, turn_count) -> float:
    next_pos = get_next_position(head, direction, h, w)
    blocked = set(my_trail) | set(opp_trail)

    score = 0.0

    # 1) immediate death check
    if next_pos in blocked:
        return -9999.0

    # 2) our reachable space from that cell
    my_space = flood_fill(next_pos, blocked, h, w, cap=MAX_FLOOD)
    my_space_size = len(my_space)
    score += my_space_size * 1.8  # important on tron

    # 3) estimate opponent space (territory differential)
    if opp_trail:
        opp_head = opp_trail[-1]
        opp_space = flood_fill(opp_head, blocked | {next_pos}, h, w, cap=MAX_FLOOD)
        opp_space_size = len(opp_space)
        territory_diff = my_space_size - opp_space_size
        score += territory_diff * 0.8

        # 4) distance to opponent (toroidal)
        dist_to_opp = manhattan_torus(next_pos, opp_head, h, w)
        # if we have more space, we can pressure
        space_adv = strategy_state.get("space_advantage", 0.0)
        if space_adv > 0.15:
            # prefer being closer but not bumping heads
            score += max(0, 10 - abs(dist_to_opp - 4))
        else:
            # prefer staying moderately away
            score += max(0, 8 - abs(dist_to_opp - 6))

        # 5) trap / enclosure potential
        trap_bonus = evaluate_trap(next_pos, my_trail, opp_trail, h, w)
        score += trap_bonus

        # --- optional: self-trap strategic bonus ---
        if opp_trail:
            opp_head = opp_trail[-1]
            dist_to_opp = manhattan_torus(next_pos, opp_head, h, w)
            if dist_to_opp > 5:  # safe to consider sealing
                seal_bonus = evaluate_self_trap_potential(next_pos, opp_head, my_trail, opp_trail, h, w)
                score += seal_bonus


        # 6) simulate opponent’s options after our move (very shallow)
        opp_dir = get_current_direction(opp_trail)
        opp_safe_after = get_safe_moves(opp_head, opp_dir, opp_trail, my_trail + [next_pos], h, w)
        if len(opp_safe_after) <= 1:
            score += 14.0  # we hurt their mobility
        elif len(opp_safe_after) == 2:
            score += 6.0

    # 7) our future escapes
    future_moves = get_safe_moves(next_pos, direction, my_trail + [next_pos], opp_trail, h, w)
    if not future_moves:
        score -= 80  # dead end
    elif len(future_moves) == 1:
        score -= 15
    elif len(future_moves) == 2:
        score += 3
    else:
        score += 10

    # 8) short lookahead path quality
    score += lookahead_quality(next_pos, direction, my_trail, opp_trail, h, w)

    # 9) repetition / pattern breaking
    recent = list(strategy_state["move_history"])
    if len(recent) >= 4 and recent[-1] == direction == recent[-2] == recent[-3]:
        score -= 9  # avoid being too predictable

    # 10) game phase bonuses
    if turn_count < 40:
        # expansion phase: reward having many reachable cells
        score += my_space_size * 0.3
    elif turn_count > 220:
        # closing phase: reward limiting opponent
        score += strategy_state.get("danger_level", 0.0) * 10

    return score


def lookahead_quality(pos, direction, my_trail, opp_trail, h, w, depth=3) -> float:
    """
    Tiny forward roll to prefer branches that stay mobile.
    """
    score = 0.0
    cur_pos = pos
    cur_dir = direction
    cur_my_trail = my_trail + [pos]

    for d in range(depth):
        moves = get_safe_moves(cur_pos, cur_dir, cur_my_trail, opp_trail, h, w)
        if not moves:
            score -= (depth - d) * 10
            break
        score += len(moves) * 2
        # pick the move with the most local space
        best_m = None
        best_s = -1
        for m in moves:
            nxt = get_next_position(cur_pos, m, h, w)
            space = flood_fill(nxt, set(cur_my_trail) | set(opp_trail), h, w, cap=80)
            if len(space) > best_s:
                best_s = len(space)
                best_m = m
        cur_pos = get_next_position(cur_pos, best_m, h, w)
        cur_dir = best_m
        cur_my_trail.append(cur_pos)

    return score


def evaluate_trap(my_next_pos, my_trail, opp_trail, h, w) -> float:
    """
    Check if our move shrinks opponent's reachable area a lot.
    """
    if not opp_trail:
        return 0.0

    opp_head = opp_trail[-1]
    blocked_now = set(my_trail) | set(opp_trail)
    # space before
    before = flood_fill(opp_head, blocked_now, h, w, cap=MAX_FLOOD)
    # space after adding our next
    after = flood_fill(opp_head, blocked_now | {my_next_pos}, h, w, cap=MAX_FLOOD)

    reduction = len(before) - len(after)
    if reduction <= 0:
        return 0.0

    bonus = reduction * 1.5
    if len(after) < 25:
        bonus += 20
    elif len(after) < 50:
        bonus += 10
    return bonus


# ---------------------------------------------------------------------
# ANALYSIS STEP (RUN EVERY TURN)
# ---------------------------------------------------------------------
def analyze_game_state():
    player_num = game_state.get("player_number", 1)
    my_trail, _, opp_trail = get_player_info(player_num)

    if not my_trail or not opp_trail:
        return

    board = game_state.get("board", {})
    h = board.get("height", BOARD_HEIGHT)
    w = board.get("width", BOARD_WIDTH)

    my_head = my_trail[-1]
    opp_head = opp_trail[-1]
    blocked = set(my_trail) | set(opp_trail)

    my_space = flood_fill(my_head, blocked, h, w, cap=MAX_FLOOD)
    opp_space = flood_fill(opp_head, blocked, h, w, cap=MAX_FLOOD)

    total = len(my_space) + len(opp_space)
    if total > 0:
        strategy_state["space_advantage"] = (len(my_space) - len(opp_space)) / total

    # danger: use immediate mobility
    my_dir = get_current_direction(my_trail)
    safe_moves = get_safe_moves(my_head, my_dir, my_trail, opp_trail, h, w)
    # 0 safe -> 1.0 danger, 3 safe -> low danger
    strategy_state["danger_level"] = max(0.0, 1.0 - (len(safe_moves) / 3.0))

    # opponent aggression: are they generally getting closer?
    opp_len = len(opp_trail)
    if opp_len >= 5:
        dist_changes = []
        for i in range(1, 5):
            d_prev = manhattan_torus(opp_trail[-i - 1], my_head, h, w)
            d_now = manhattan_torus(opp_trail[-i], my_head, h, w)
            dist_changes.append(d_prev - d_now)
        avg = sum(dist_changes) / len(dist_changes)
        strategy_state["opponent_aggression"] = max(0.0, min(1.0, 0.5 + avg / 3.0))


# ---------------------------------------------------------------------
# BOOST DECISION
# ---------------------------------------------------------------------
def add_boost_decision(move: str, boosts_left: int, turn_count: int) -> str:
    if boosts_left <= 0:
        return move

    danger = strategy_state.get("danger_level", 0.0)
    space_adv = strategy_state.get("space_advantage", 0.0)

    use = False
    # highest priority: escape
    if danger > 0.7:
        use = True
    # midgame push if behind
    elif 40 < turn_count < 200 and space_adv < -0.25 and random.random() < 0.35:
        use = True
    # very late: dump boosts
    elif turn_count > 360 and boosts_left > 0:
        use = True

    if use:
        strategy_state["boost_used_turns"].append(turn_count)
        return f"{move}:BOOST"
    return move


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------


def get_player_info(player_number: int):
    if player_number == 1:
        return (
            game_state.get("agent1_trail", []),
            game_state.get("agent1_boosts", 3),
            game_state.get("agent2_trail", []),
        )
    else:
        return (
            game_state.get("agent2_trail", []),
            game_state.get("agent2_boosts", 3),
            game_state.get("agent1_trail", []),
        )
# --- Add these near your other utilities ---

def players_are_separated(my_head, opp_head, blocked, h, w, turn_count, min_turn=35) -> bool:
    """
    Returns True if opponent is unreachable from our region.
    - Don't count heads as blocked, or we will *always* say separated.
    - Also wait until a few turns have passed so early game still does space-conquer.
    """
    if turn_count < min_turn:
        return False  # force space-conquering early

    # make a copy of blocked but allow heads
    relaxed_blocked = set(blocked)
    if my_head in relaxed_blocked:
        relaxed_blocked.remove(my_head)
    if opp_head in relaxed_blocked:
        relaxed_blocked.remove(opp_head)

    my_region = flood_fill(my_head, relaxed_blocked, h, w, cap=9999)
    return opp_head not in my_region



def efficient_path_move(head, current_dir, my_trail, h, w) -> str:
    """
    Wall-following traversal (right-hand rule) that avoids entering 1-cell pockets.
    """
    dirs = ["UP", "RIGHT", "DOWN", "LEFT"]
    dir_idx = dirs.index(current_dir)
    blocked = set(my_trail)

    # define priority order (forward → right → left → back)
    forward = dirs[dir_idx]
    right = dirs[(dir_idx + 1) % 4]
    left = dirs[(dir_idx - 1) % 4]
    back = dirs[(dir_idx + 2) % 4]
    order = [forward, right, left, back]

    # helper: avoid “1-gap” dead ends (check next two cells)
    def safe_two_steps(direction):
        first = get_next_position(head, direction, h, w)
        second = get_next_position(first, direction, h, w)
        # both cells must be free
        return first not in blocked and second not in blocked

    # first pass: prefer two-free-step paths
    for d in order:
        if safe_two_steps(d):
            return d

    # fallback: allow single-safe-step
    for d in order:
        nxt = get_next_position(head, d, h, w)
        if nxt not in blocked:
            return d

    return forward  # worst case fallback


def get_current_direction(trail: List[Tuple[int, int]]) -> str:
    if len(trail) < 2:
        return "RIGHT"

    board = game_state.get("board", {})
    w = board.get("width", BOARD_WIDTH)
    h = board.get("height", BOARD_HEIGHT)

    x2, y2 = trail[-1]
    x1, y1 = trail[-2]

    dx = x2 - x1
    dy = y2 - y1

    # adjust for wrap: pick smaller step
    if abs(dx) > w // 2:
        dx = -1 if dx > 0 else 1
    if abs(dy) > h // 2:
        dy = -1 if dy > 0 else 1

    if dx == 1:
        return "RIGHT"
    if dx == -1:
        return "LEFT"
    if dy == 1:
        return "DOWN"
    if dy == -1:
        return "UP"
    return "RIGHT"


def get_next_position(pos: Tuple[int, int], direction: str, h: int, w: int) -> Tuple[int, int]:
    x, y = pos
    if direction == "UP":
        return (x, (y - 1) % h)
    if direction == "DOWN":
        return (x, (y + 1) % h)
    if direction == "LEFT":
        return ((x - 1) % w, y)
    if direction == "RIGHT":
        return ((x + 1) % w, y)
    return pos


def is_move_safe(head: Tuple[int, int], direction: str, my_trail, opp_trail) -> bool:
    board = game_state.get("board", {})
    h = board.get("height", BOARD_HEIGHT)
    w = board.get("width", BOARD_WIDTH)
    nxt = get_next_position(head, direction, h, w)
    return nxt not in my_trail and nxt not in opp_trail


def get_safe_moves(head, current_dir, my_trail, opp_trail, h, w) -> List[str]:
    safe = []
    for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
        if d == opposite_dir(current_dir):
            continue
        nxt = get_next_position(head, d, h, w)
        if nxt not in my_trail and nxt not in opp_trail:
            safe.append(d)
    return safe


def opposite_dir(d: str) -> str:
    return {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}.get(d, "")


def flood_fill(start: Tuple[int, int], blocked: Set[Tuple[int, int]],
               h: int, w: int, cap: int = 9999) -> Set[Tuple[int, int]]:
    """
    Torus-aware BFS reachability with cap.
    """
    if start in blocked:
        return set()

    visited = {start}
    q = deque([start])

    while q and len(visited) < cap:
        x, y = q.popleft()
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            nx, ny = get_next_position((x, y), d, h, w)
            if (nx, ny) not in visited and (nx, ny) not in blocked:
                visited.add((nx, ny))
                q.append((nx, ny))

    return visited


def manhattan_torus(a: Tuple[int, int], b: Tuple[int, int], h: int, w: int) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dx = min(dx, w - dx)
    dy = min(dy, h - dy)
    return dx + dy

def evaluate_self_trap_potential(my_head, opp_head, my_trail, opp_trail, h, w) -> float:
    """
    Returns a positive score bonus if closing the region would give us a lasting advantage.
    Used as a soft heuristic, not a binary trigger.
    """
    blocked = set(my_trail) | set(opp_trail)
    my_region = flood_fill(my_head, blocked, h, w, cap=9999)
    opp_region = flood_fill(opp_head, blocked, h, w, cap=9999)

    space_ratio = len(my_region) / max(1, len(opp_region))
    if space_ratio < 1.1:
        return 0.0  # not dominant enough

    # find border area overlap
    border = find_border_between_regions(my_region, opp_region, h, w)
    if not border:
        return 0.0

    # test sealing each border cell (virtual closure)
    best_gain = 0
    for b in border:
        new_blocked = blocked | {b}
        new_my = flood_fill(my_head, new_blocked, h, w, cap=9999)
        new_opp = flood_fill(opp_head, new_blocked, h, w, cap=9999)
        gain = (len(new_my) - len(my_region)) - (len(new_opp) - len(opp_region))
        best_gain = max(best_gain, gain)

    # convert to score scaling
    if best_gain > 0:
        return min(best_gain * 0.5, 60.0)  # max bonus cap
    return 0.0

def should_self_trap(my_head, opp_head, my_trail, opp_trail, h, w) -> bool:
    """
    Returns True if closing the frontier would result in us keeping
    significantly more space than the opponent.
    """
    blocked = set(my_trail) | set(opp_trail)
    my_space = flood_fill(my_head, blocked, h, w, cap=9999)
    opp_space = flood_fill(opp_head, blocked, h, w, cap=9999)
    if not my_space or not opp_space:
        return False

    border_cells = find_border_between_regions(my_space, opp_space, h, w)
    if not border_cells:
        return False

    for cell in border_cells:
        test_blocked = blocked | {cell}
        my_new = flood_fill(my_head, test_blocked, h, w, cap=9999)
        opp_new = flood_fill(opp_head, test_blocked, h, w, cap=9999)
        # keep if we still have ~15% more region after sealing
        if len(my_new) > len(opp_new) * 1.15:
            return True

    return False

def find_sealing_move(my_head, my_trail, opp_trail, h, w) -> str:
    """
    Picks the direction that most reduces the opponent's reachable space.
    Returns the best sealing move or None if none help.
    """
    dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
    blocked = set(my_trail) | set(opp_trail)
    best_move = None
    best_reduction = 0

    opp_head = opp_trail[-1] if opp_trail else None
    if not opp_head:
        return None

    before = flood_fill(opp_head, blocked, h, w, cap=9999)

    for d in dirs:
        nxt = get_next_position(my_head, d, h, w)
        if nxt in blocked:
            continue
        after = flood_fill(opp_head, blocked | {nxt}, h, w, cap=9999)
        reduction = len(before) - len(after)
        if reduction > best_reduction:
            best_reduction = reduction
            best_move = d

    return best_move


def find_border_between_regions(region1, region2, h, w):
    """
    Return approximate touching frontier between two flood-fill regions.
    """
    borders = set()
    for (x, y) in region1:
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            nx, ny = get_next_position((x, y), d, h, w)
            if (nx, ny) in region2:
                borders.add((x, y))
    return borders



# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on {port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
