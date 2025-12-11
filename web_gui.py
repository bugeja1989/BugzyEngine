#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
web_gui.py: Web GUI for the BugzyEngine.
"""

from flask import Flask, render_template_string, request, redirect, url_for
import chess
import chess.svg
import torch
import os

from neural_network.src.engine_utils import alpha_beta_search
from process2_trainer import ChessNet # Import model class
from config import WEB_PORT, GPU_DEVICE

app = Flask(__name__)
board = chess.Board()
device = torch.device(GPU_DEVICE)

# --- Load Model ---
MODEL_PATH = "/home/ubuntu/BugzyEngine/models/bugzy_model.pth"
model = ChessNet().to(device)
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set to evaluation mode
else:
    print("No model found. The engine will play randomly.")
    model = None

@app.route("/")
def index():
    board_svg = chess.svg.board(board=board, size=400)
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BugzyEngine</title>
            <meta http-equiv="refresh" content="5"> <!-- Auto-refresh every 5 seconds -->
            <style>
                body { font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column; background-color: #333; color: #fff; }
                .board { margin-bottom: 20px; }
                form { display: flex; flex-direction: column; align-items: center; }
                input[type="text"] { width: 200px; padding: 8px; margin-bottom: 10px; border-radius: 4px; border: 1px solid #ccc; }
                input[type="submit"] { padding: 10px 20px; border: none; border-radius: 4px; background-color: #4CAF50; color: white; cursor: pointer; }
                a { color: #4CAF50; margin-top: 20px; }
                h1, h2 { text-align: center; }
                .game-over { font-size: 24px; color: #ff5c5c; }
            </style>
        </head>
        <body>
            <h1>BugzyEngine</h1>
            <div class="board">{{ board_svg | safe }}</div>
            {% if game_over %}
                <h2 class="game-over">Game Over: {{ result }}</h2>
            {% else %}
                <form action="/move" method="get">
                    <label for="move">Your move (e.g., e2e4):</label>
                    <input type="text" id="move" name="move" autofocus>
                    <input type="submit" value="Make Move">
                </form>
            {% endif %}
            <a href="/reset">New Game</a>
        </body>
        </html>
    """, board_svg=board_svg, game_over=board.is_game_over(), result=board.result())

@app.route("/move")
def move():
    if board.is_game_over():
        return redirect(url_for('index'))

    user_move_str = request.args.get("move")
    if user_move_str:
        try:
            move = chess.Move.from_uci(user_move_str)
            if move in board.legal_moves:
                board.push(move)
                # Engine's turn
                if not board.is_game_over():
                    print("BugzyEngine is thinking...")
                    _, engine_move = alpha_beta_search(board, depth=3, model=model)
                    if engine_move:
                        print(f"BugzyEngine plays: {engine_move}")
                        board.push(engine_move)
                    else:
                        print("Engine has no legal moves!")
            else:
                print(f"Invalid move: {user_move_str}")
        except ValueError:
            print(f"Invalid move format: {user_move_str}")
            pass # Invalid move format
    return redirect(url_for('index'))

@app.route("/reset")
def reset():
    board.reset()
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=WEB_PORT, debug=True)
