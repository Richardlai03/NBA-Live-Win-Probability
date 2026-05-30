import os
import sys
import threading
import time
import numpy as np
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit

sys.path.append(os.path.join(os.path.dirname(__file__), "../model"))
from net import WinProbNet, FEATURE_NAMES
from live_feed import get_current_game_state, get_live_game_ids, get_historical_game_state

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.pt")
MEAN_PATH  = os.path.join(os.path.dirname(__file__), "../model/feature_mean.npy")
STD_PATH   = os.path.join(os.path.dirname(__file__), "../model/feature_std.npy")

# seconds between nba_api polls
POLL_INTERVAL = 15  

app       = Flask(__name__)
app.config["SECRET_KEY"] = "nba-win-prob"
socketio  = SocketIO(app, cors_allowed_origins="*")

# Load model once at startup
model = WinProbNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
feature_mean = np.load(MEAN_PATH)
feature_std  = np.load(STD_PATH)

# Track active polling threads so we don't spawn duplicates
active_games: set[str] = set()
active_games_lock = threading.Lock()

def predict(game_state: dict) -> float:
    x = np.array([game_state[f] for f in FEATURE_NAMES], dtype=np.float32)
    x = (x - feature_mean) / (feature_std + 1e-8)
    with torch.no_grad():
        logit = model(torch.from_numpy(x).unsqueeze(0))
        return torch.sigmoid(logit).item()


def poll_game(game_id: str):
    print(f"[{game_id}] Polling started")
    while True:
        state = get_current_game_state(game_id)

        # replay historical game
        if state is None:
            from live_feed import get_historical_game_state
            state = get_historical_game_state(game_id)

        if state is None:
            # API error: wait and retry
            time.sleep(POLL_INTERVAL)
            continue

        prob = predict(state)

        socketio.emit("update", {
            "game_id":          game_id,
            "home_prob":        round(prob, 4),
            "away_prob":        round(1 - prob, 4),
            "home_score":       state["home_score"],
            "away_score":       state["away_score"],
            "score_diff":       state["score_diff"],
            "time_remaining_s": state["time_remaining_s"],
            "period":           state["period"],
            "last_play":        state.get("last_play", ""),
        })

        # Stop polling when game is final (period > 4 and clock at 0, or time_remaining_s == 0 in regulation)
        if state["period"] >= 4 and state["time_remaining_s"] == 0:
            print(f"[{game_id}] Game final — stopping poll")
            with active_games_lock:
                active_games.discard(game_id)
            break

        time.sleep(POLL_INTERVAL)


@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/live_games")
def live_games():
    return jsonify(get_live_game_ids())

@socketio.on("start_game")
def on_start_game(data):
    game_id = data.get("game_id", "").strip()
    if not game_id:
        return

    with active_games_lock:
        if game_id in active_games:
            print(f"[{game_id}] Already polling — ignoring duplicate request")
            return
        active_games.add(game_id)

    t = threading.Thread(target=poll_game, args=(game_id,), daemon=True)
    t.start()


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)