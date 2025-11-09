"""
Improved Case Closed Agent — Flask-based
Avoids crashing into trails and makes basic safe decisions.
"""

import os
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "FernandoG")
AGENT_NAME = os.getenv("AGENT_NAME", "SafeAgent")

# Game state tracker
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    """Health check for the judge."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive and store the latest game state from the judge."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    game_state.update(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Return a move decision to the judge."""
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = game_state.get("turn_count", 0)

    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])

    move = decide_move(my_trail, other_trail, turn_count, my_boosts)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Notify the agent the game has ended."""
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


# 🧩 --- Smarter Decision Logic --- 🧩
def decide_move(my_trail, other_trail, turn_count, my_boosts):
    """Avoids trails, prefers safe moves, uses boosts sparingly."""
    if not my_trail:
        return "RIGHT"

    head = tuple(my_trail[-1])

    # Get direction of last move
    if len(my_trail) >= 2:
        prev = tuple(my_trail[-2])
        dx, dy = head[0] - prev[0], head[1] - prev[1]
    else:
        dx, dy = 1, 0  # Default right

    # Map directions to (dx, dy)
    directions = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    opposite = {
        "UP": "DOWN",
        "DOWN": "UP",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
    }

    # Identify current direction
    current_dir = "RIGHT"
    for name, vec in directions.items():
        if vec == (dx, dy):
            current_dir = name
            break

    # All occupied cells
    obstacles = set(tuple(p) for p in (my_trail + other_trail))

    # Check which moves are safe
    safe_moves = []
    for dname, (ddx, ddy) in directions.items():
        if dname == opposite[current_dir]:
            continue
        new_head = (head[0] + ddx, head[1] + ddy)
        if new_head not in obstacles:
            safe_moves.append(dname)

    # Choose move
    if current_dir in safe_moves:
        move = current_dir
    elif safe_moves:
        move = random.choice(safe_moves)
    else:
        move = current_dir  # no escape

    # Optional: controlled boost use
    use_boost = my_boosts > 0 and 30 <= turn_count <= 80 and random.random() < 0.25
    return f"{move}:BOOST" if use_boost else move


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"🚀 Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
