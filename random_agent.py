import os
from collections import deque
from threading import Lock

from flask import Flask, jsonify, request

from case_closed_game import Direction, Game, AGENT
import random

app = Flask(__name__)

game_lock = Lock()
GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

PARTICIPANT = "ParticipantX"
AGENT_NAME = "RandomBot"


def _update_local_game_from_post(data: dict):
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2

    mapping = {Direction.UP: "UP", Direction.DOWN: "DOWN", Direction.LEFT: "LEFT", Direction.RIGHT: "RIGHT"}
    current_dir = my_agent.direction
    moves = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]

    def will_collide(direction: Direction) -> bool:
        head_x, head_y = my_agent.trail[-1]
        dx, dy = direction.value
        width = GLOBAL_GAME.board.width
        height = GLOBAL_GAME.board.height
        nx = (head_x + dx) % width
        ny = (head_y + dy) % height
        return GLOBAL_GAME.board.grid[ny][nx] == AGENT

    safe_moves = []

    for direction in moves:
        if direction.value == (-current_dir.value[0], -current_dir.value[1]):
            continue
        if will_collide(direction):
            continue
        safe_moves.append(direction)

    if not safe_moves:
        for direction in moves:
            if direction.value == (-current_dir.value[0], -current_dir.value[1]):
                continue
            safe_moves.append(direction)

    if not safe_moves:
        safe_moves = moves

    chosen = random.choice(safe_moves)
    return jsonify({"move": mapping[chosen]}), 200


@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
