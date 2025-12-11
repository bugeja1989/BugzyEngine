#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
web_gui.py: Cyberpunk Web GUI for the BugzyEngine with drag-and-drop and enhanced features.
"""

from flask import Flask, render_template_string, request, jsonify
import chess
import chess.pgn
import torch
import os
import json
from datetime import datetime

from neural_network.src.engine_utils import alpha_beta_search
from process2_trainer import ChessNet
from config import WEB_PORT, GPU_DEVICE

app = Flask(__name__)
board = chess.Board()
device = torch.device(GPU_DEVICE)
move_history = []
engine_logs = []
game_start_time = None

# --- Hot-Reload Model System ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bugzy_model.pth")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "processed")
model = ChessNet().to(device)
model_loaded = False
model_last_modified = 0
model_version = 0

def load_model():
    """Load or reload the model if it has been updated."""
    global model, model_loaded, model_last_modified, model_version
    
    if not os.path.exists(MODEL_PATH):
        return False
    
    current_mtime = os.path.getmtime(MODEL_PATH)
    
    # Check if model has been updated
    if current_mtime > model_last_modified:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            model_loaded = True
            model_last_modified = current_mtime
            model_version += 1
            add_log(f"🔄 Model reloaded! Version: {model_version}")
            print(f"[HOT-RELOAD] Model updated to version {model_version}")
            return True
        except Exception as e:
            print(f"[HOT-RELOAD] Error loading model: {e}")
            return False
    
    return model_loaded

# Initial model load
if load_model():
    print(f"Model loaded successfully. Version: {model_version}")
else:
    print("No model found. The engine will play randomly.")

def count_trained_positions():
    """Count total positions from processed PGN files."""
    if not os.path.exists(DATA_PATH):
        return 0
    count = 0
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pgn"):
            try:
                with open(os.path.join(DATA_PATH, filename)) as f:
                    pgn = chess.pgn.read_game(f)
                    if pgn:
                        count += len(list(pgn.mainline_moves()))
            except:
                pass
    return count

def add_log(message):
    """Add a log entry with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    engine_logs.append(f"[{timestamp}] {message}")
    if len(engine_logs) > 50:  # Keep last 50 logs
        engine_logs.pop(0)

@app.route("/")
def index():
    return render_template_string(CYBERPUNK_HTML)

@app.route("/api/board")
def api_board():
    """Return current board state as FEN."""
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "turn": "white" if board.turn == chess.WHITE else "black"
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    """Handle player move and return engine response."""
    data = request.json
    move_uci = data.get("move")
    
    if not move_uci:
        return jsonify({"error": "No move provided"}), 400
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"error": "Illegal move"}), 400
        
        # Player's move - get SAN before pushing
        san_move = board.san(move)
        board.push(move)
        move_history.append({"move": san_move, "player": "human"})
        add_log(f"👤 Player: {san_move}")
        
        # Check if game over
        if board.is_game_over():
            add_log(f"🏁 Game Over: {board.result()}")
            return jsonify({
                "fen": board.fen(),
                "game_over": True,
                "result": board.result(),
                "engine_move": None,
                "san_move": san_move
            })
        
        # Engine's turn - check for model updates first!
        load_model()  # Hot-reload if model was updated
        add_log("🤖 BugzyEngine thinking...")
        _, engine_move = alpha_beta_search(board, depth=3, model=model if model_loaded else None)
        
        if engine_move:
            san_engine_move = board.san(engine_move)
            board.push(engine_move)
            move_history.append({"move": san_engine_move, "player": "engine"})
            add_log(f"⚡ BugzyEngine: {san_engine_move} [CHAOS MODE]")
            
            return jsonify({
                "fen": board.fen(),
                "game_over": board.is_game_over(),
                "result": board.result() if board.is_game_over() else None,
                "engine_move": engine_move.uci(),
                "san_move": san_move,
                "san_engine_move": san_engine_move
            })
        else:
            add_log("❌ No legal moves available!")
            return jsonify({"error": "No legal moves"}), 500
            
    except Exception as e:
        add_log(f"⚠️ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the board."""
    global move_history, game_start_time
    board.reset()
    move_history = []
    game_start_time = datetime.now()
    add_log("🔄 New game started!")
    return jsonify({"fen": board.fen()})

@app.route("/api/stats")
def api_stats():
    """Return engine statistics."""
    positions_trained = count_trained_positions()
    gpu_status = "METAL ACTIVE" if device.type == "mps" else device.type.upper()
    
    # Calculate game time
    game_time = "00:00"
    if game_start_time:
        elapsed = datetime.now() - game_start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        game_time = f"{minutes:02d}:{seconds:02d}"
    
    # Check for model updates
    load_model()
    
    return jsonify({
        "positions_trained": positions_trained,
        "gpu_status": gpu_status,
        "model_loaded": model_loaded,
        "model_version": model_version,
        "move_count": len(move_history),
        "game_time": game_time
    })

@app.route("/api/logs")
def api_logs():
    """Return recent engine logs."""
    return jsonify({"logs": engine_logs})

@app.route("/api/pgn")
def api_pgn():
    """Return current game as PGN."""
    game = chess.pgn.Game()
    game.headers["Event"] = "BugzyEngine vs Human"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Human"
    game.headers["Black"] = "BugzyEngine"
    
    node = game
    temp_board = chess.Board()
    for i, move_data in enumerate(move_history):
        # Find the move from SAN
        for legal_move in temp_board.legal_moves:
            if temp_board.san(legal_move) == move_data["move"]:
                node = node.add_variation(legal_move)
                temp_board.push(legal_move)
                break
    
    return jsonify({"pgn": str(game)})

# Cyberpunk HTML Template with Wikipedia piece images
CYBERPUNK_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BugzyEngine 3.0 - Cyberpunk Chess</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff00;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        h1 {
            text-align: center;
            font-size: 3em;
            color: #00ffff;
            text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff;
            margin-bottom: 30px;
            letter-spacing: 5px;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 600px 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #00ffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }
        
        .panel h2 {
            color: #ff00ff;
            text-shadow: 0 0 10px #ff00ff;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        #board {
            width: 600px;
            margin: 0 auto;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        .stat-box {
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00ffff;
            padding: 10px;
            border-radius: 5px;
        }
        
        .stat-label {
            color: #888;
            font-size: 0.8em;
        }
        
        .stat-value {
            color: #00ffff;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .btn {
            background: linear-gradient(135deg, #00ffff 0%, #0088ff 100%);
            border: none;
            color: #000;
            padding: 12px 24px;
            font-size: 1em;
            font-weight: bold;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin-bottom: 10px;
        }
        
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ffff;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff0055 0%, #ff5500 100%);
        }
        
        .logs-container {
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #00ff00;
            border-radius: 5px;
            padding: 10px;
            height: 400px;
            overflow-y: auto;
            font-size: 0.9em;
        }
        
        .log-entry {
            color: #00ff00;
            margin-bottom: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff00;
            box-shadow: 0 0 10px #00ff00;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .game-status {
            text-align: center;
            font-size: 1.5em;
            padding: 15px;
            background: rgba(255, 0, 85, 0.3);
            border: 2px solid #ff0055;
            border-radius: 5px;
            margin-top: 15px;
            color: #ff0055;
            text-shadow: 0 0 10px #ff0055;
        }
        
        .move-history {
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid #ff00ff;
            border-radius: 5px;
            padding: 10px;
            height: 250px;
            overflow-y: auto;
        }
        
        .move-entry {
            color: #ff00ff;
            margin-bottom: 3px;
            padding: 3px;
        }
        
        .move-entry.human {
            color: #00ffff;
        }
        
        .move-entry.engine {
            color: #ff0055;
        }
        
        .turn-indicator {
            text-align: center;
            padding: 10px;
            background: rgba(0, 255, 255, 0.2);
            border: 1px solid #00ffff;
            border-radius: 5px;
            margin-bottom: 10px;
            color: #00ffff;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ BUGZYENGINE 3.0 ⚡</h1>
        
        <div class="main-grid">
            <!-- Left Panel: Stats & Controls -->
            <div class="panel">
                <h2>🎮 CONTROL PANEL</h2>
                <button class="btn" onclick="resetGame()">🔄 NEW GAME</button>
                <button class="btn btn-danger" onclick="copyPGN()">📋 COPY PGN</button>
                
                <h2 style="margin-top: 20px;">📊 STATS</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">GPU STATUS</div>
                        <div class="stat-value" id="gpu-status">
                            <span class="status-indicator"></span> LOADING...
                        </div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">POSITIONS TRAINED</div>
                        <div class="stat-value" id="positions-trained">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MODEL STATUS</div>
                        <div class="stat-value" id="model-status">LOADING...</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MOVE COUNT</div>
                        <div class="stat-value" id="move-count">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">GAME TIME</div>
                        <div class="stat-value" id="game-time">00:00</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">DEVICE</div>
                        <div class="stat-value" id="device">MPS</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MODEL VERSION</div>
                        <div class="stat-value" id="model-version">0</div>
                    </div>
                </div>
                
                <h2 style="margin-top: 20px;">📜 MOVE HISTORY</h2>
                <div class="move-history" id="move-history"></div>
            </div>
            
            <!-- Center Panel: Chess Board -->
            <div class="panel">
                <div class="turn-indicator" id="turn-indicator">White to move</div>
                <div id="board"></div>
                <div id="game-status" class="game-status" style="display: none;"></div>
            </div>
            
            <!-- Right Panel: Logs -->
            <div class="panel">
                <h2>🔥 ENGINE LOGS</h2>
                <div class="logs-container" id="logs"></div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script>
        let board = null;
        let game_over = false;
        
        function onDragStart(source, piece, position, orientation) {
            if (game_over) return false;
            if (piece.search(/^b/) !== -1) return false; // Only allow white pieces
        }
        
        function onDrop(source, target) {
            const move = source + target;
            
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({move: move})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return 'snapback';
                }
                
                board.position(data.fen);
                updateGameStatus(data);
                updateStats();
                updateLogs();
                updateMoveHistory();
            })
            .catch(error => {
                console.error('Error:', error);
                return 'snapback';
            });
        }
        
        function updateGameStatus(data) {
            const statusDiv = document.getElementById('game-status');
            const turnDiv = document.getElementById('turn-indicator');
            
            if (data.game_over) {
                statusDiv.style.display = 'block';
                statusDiv.textContent = `🏁 GAME OVER: ${data.result}`;
                turnDiv.textContent = 'Game Over';
                game_over = true;
            } else {
                statusDiv.style.display = 'none';
                turnDiv.textContent = data.fen.includes(' w ') ? 'White to move' : 'Black to move';
                game_over = false;
            }
        }
        
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('gpu-status').innerHTML = 
                        `<span class="status-indicator"></span> ${data.gpu_status}`;
                    document.getElementById('positions-trained').textContent = 
                        data.positions_trained.toLocaleString();
                    document.getElementById('model-status').textContent = 
                        data.model_loaded ? 'LOADED' : 'RANDOM';
                    document.getElementById('move-count').textContent = data.move_count;
                    document.getElementById('game-time').textContent = data.game_time;
                    document.getElementById('model-version').textContent = data.model_version;
                });
        }
        
        function updateLogs() {
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    const logsDiv = document.getElementById('logs');
                    logsDiv.innerHTML = data.logs.map(log => 
                        `<div class="log-entry">${log}</div>`
                    ).join('');
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                });
        }
        
        function updateMoveHistory() {
            fetch('/api/board')
                .then(response => response.json())
                .then(data => {
                    // Move history is updated via stats
                });
        }
        
        function resetGame() {
            fetch('/api/reset', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    board.position(data.fen);
                    game_over = false;
                    document.getElementById('game-status').style.display = 'none';
                    document.getElementById('turn-indicator').textContent = 'White to move';
                    document.getElementById('move-history').innerHTML = '';
                    updateStats();
                    updateLogs();
                });
        }
        
        function copyPGN() {
            fetch('/api/pgn')
                .then(response => response.json())
                .then(data => {
                    navigator.clipboard.writeText(data.pgn);
                    alert('✅ PGN copied to clipboard!');
                });
        }
        
        // Initialize board with Wikipedia piece images
        const config = {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };
        
        board = Chessboard('board', config);
        
        // Update stats and logs every 2 seconds
        setInterval(() => {
            updateStats();
            updateLogs();
        }, 2000);
        
        // Initial load
        updateStats();
        updateLogs();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    game_start_time = datetime.now()
    add_log("⚡ BugzyEngine 3.0 started!")
    add_log(f"🖥️  GPU Device: {device}")
    add_log(f"🧠 Model loaded: {model_loaded}")
    app.run(host='0.0.0.0', port=WEB_PORT, debug=True)
