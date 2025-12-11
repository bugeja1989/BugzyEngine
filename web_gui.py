#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
web_gui_v5.py: BugzyEngine v5.0 - Complete Game Mode System
Features:
- Manual Mode: You vs BugzyEngine with move suggestions
- Simulation Mode: BugzyEngine vs Stockfish (auto-play)
- Color Selection: Choose White/Black
- Stockfish ELO: 500, 1000, 1500, 2000, 2500, 3000, 3200+
- Move Suggestions: Real-time best moves
- Enhanced Cyberpunk GUI
"""

from flask import Flask, render_template_string, request, jsonify
import chess
import chess.engine
import chess.pgn
import torch
import os
import json
import threading
import time
from datetime import datetime

from neural_network.src.engine_utils import alpha_beta_search
from process2_trainer import ChessNet
from config import WEB_PORT, GPU_DEVICE

app = Flask(__name__)

# --- Global State ---
board = chess.Board()
device = torch.device(GPU_DEVICE)
move_history = []
engine_logs = []
game_start_time = None

# Game Mode State
game_mode = "manual"  # "manual" or "simulation"
player_color = "white"  # "white" or "black"
stockfish_elo = 2000
simulation_running = False
simulation_thread = None

# Stockfish Engine
stockfish_engine = None
STOCKFISH_PATH = "stockfish"  # Assumes stockfish is in PATH

# --- Model Management ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bugzy_model.pth")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "processed")
model = ChessNet().to(device)
model_loaded = False
model_last_modified = 0
model_version = 0

def add_log(message):
    """Add a log entry with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    engine_logs.append(f"[{timestamp}] {message}")
    if len(engine_logs) > 50:
        engine_logs.pop(0)

def load_model():
    """Load or reload the model if it has been updated."""
    global model, model_loaded, model_last_modified, model_version
    
    if not os.path.exists(MODEL_PATH):
        return False
    
    current_mtime = os.path.getmtime(MODEL_PATH)
    
    if current_mtime > model_last_modified:
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            model_loaded = True
            model_last_modified = current_mtime
            model_version += 1
            add_log(f"🔄 Model reloaded! Version: {model_version}")
            return True
        except Exception as e:
            add_log(f"⚠️ Error loading model: {e}")
            return False
    
    return model_loaded

# Initialize Stockfish
def init_stockfish():
    """Initialize Stockfish engine."""
    global stockfish_engine
    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        add_log("✅ Stockfish initialized")
        return True
    except Exception as e:
        add_log(f"⚠️ Stockfish not found: {e}")
        return False

# Initial setup
if load_model():
    add_log(f"🧠 BugzyEngine Model v{model_version} loaded")
else:
    add_log("⚠️ No model found - playing randomly")

init_stockfish()

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

def get_stockfish_move(elo_level, time_limit=0.1):
    """Get move from Stockfish at specified ELO."""
    if not stockfish_engine:
        return None
    
    try:
        # Set ELO limit
        stockfish_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo_level})
        result = stockfish_engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    except Exception as e:
        add_log(f"⚠️ Stockfish error: {e}")
        return None

def get_bugzy_move():
    """Get move from BugzyEngine."""
    load_model()  # Hot-reload check
    _, move = alpha_beta_search(board, depth=3, model=model if model_loaded else None)
    return move

def get_move_suggestions(num_suggestions=3):
    """Get top move suggestions with evaluations."""
    suggestions = []
    
    if not stockfish_engine:
        return suggestions
    
    try:
        # Get top moves from Stockfish analysis
        info = stockfish_engine.analyse(board, chess.engine.Limit(depth=15), multipv=num_suggestions)
        
        for i, result in enumerate(info if isinstance(info, list) else [info]):
            move = result.get("pv", [None])[0]
            score = result.get("score", None)
            
            if move and score:
                # Convert score to centipawns
                if score.is_mate():
                    eval_str = f"M{score.mate()}"
                else:
                    cp = score.relative.score()
                    eval_str = f"{cp/100:+.2f}"
                
                suggestions.append({
                    "move": board.san(move),
                    "uci": move.uci(),
                    "eval": eval_str,
                    "rank": i + 1
                })
    except Exception as e:
        add_log(f"⚠️ Analysis error: {e}")
    
    return suggestions

# --- Simulation Mode ---
def run_simulation():
    """Run BugzyEngine vs Stockfish simulation."""
    global simulation_running, board, move_history
    
    add_log(f"🎮 Simulation started: BugzyEngine vs Stockfish (ELO {stockfish_elo})")
    
    while simulation_running and not board.is_game_over():
        time.sleep(1)  # Delay for visualization
        
        if board.turn == chess.WHITE:
            # BugzyEngine plays White
            move = get_bugzy_move()
            player_name = "BugzyEngine"
        else:
            # Stockfish plays Black
            move = get_stockfish_move(stockfish_elo)
            player_name = f"Stockfish-{stockfish_elo}"
        
        if move and move in board.legal_moves:
            san_move = board.san(move)
            board.push(move)
            move_history.append({"move": san_move, "player": player_name})
            add_log(f"{'⚪' if board.turn == chess.BLACK else '⚫'} {player_name}: {san_move}")
        else:
            add_log("⚠️ Invalid move in simulation")
            break
    
    if board.is_game_over():
        result = board.result()
        add_log(f"🏁 Game Over: {result}")
        
        if result == "1-0":
            add_log("🎉 BugzyEngine WINS!")
        elif result == "0-1":
            add_log(f"😔 Stockfish-{stockfish_elo} wins")
        else:
            add_log("🤝 Draw")
    
    simulation_running = False

# --- API Routes ---
@app.route("/")
def index():
    return render_template_string(CYBERPUNK_HTML_V5)

@app.route("/api/board")
def api_board():
    """Return current board state."""
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "move_history": move_history
    })

@app.route("/api/move", methods=["POST"])
def api_move():
    """Handle player move in manual mode."""
    global board, move_history, game_start_time
    
    if game_mode != "manual":
        return jsonify({"error": "Not in manual mode"}), 400
    
    if not game_start_time:
        game_start_time = datetime.now()
    
    move_uci = request.json.get("move")
    if not move_uci:
        return jsonify({"error": "No move provided"}), 400
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return jsonify({"error": "Illegal move"}), 400
        
        # Player's move
        san_move = board.san(move)
        board.push(move)
        move_history.append({"move": san_move, "player": "You"})
        add_log(f"👤 You: {san_move}")
        
        # Check if game over
        if board.is_game_over():
            add_log(f"🏁 Game Over: {board.result()}")
            return jsonify({
                "fen": board.fen(),
                "game_over": True,
                "result": board.result(),
                "engine_move": None,
                "san_move": san_move,
                "suggestions": []
            })
        
        # Engine's turn
        load_model()
        add_log("🤖 BugzyEngine thinking...")
        engine_move = get_bugzy_move()
        
        if engine_move:
            san_engine_move = board.san(engine_move)
            board.push(engine_move)
            move_history.append({"move": san_engine_move, "player": "BugzyEngine"})
            add_log(f"🤖 BugzyEngine: {san_engine_move}")
        
        # Get suggestions for next move
        suggestions = get_move_suggestions()
        
        return jsonify({
            "fen": board.fen(),
            "game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
            "engine_move": engine_move.uci() if engine_move else None,
            "san_move": san_move,
            "suggestions": suggestions
        })
    
    except Exception as e:
        add_log(f"⚠️ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/suggestions")
def api_suggestions():
    """Get move suggestions for current position."""
    suggestions = get_move_suggestions()
    return jsonify({"suggestions": suggestions})

@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update game settings."""
    global game_mode, player_color, stockfish_elo
    
    data = request.json
    if "mode" in data:
        game_mode = data["mode"]
        add_log(f"🎮 Mode: {game_mode}")
    
    if "color" in data:
        player_color = data["color"]
        add_log(f"🎨 Playing as: {player_color}")
    
    if "stockfish_elo" in data:
        stockfish_elo = int(data["stockfish_elo"])
        add_log(f"🎯 Stockfish ELO: {stockfish_elo}")
    
    return jsonify({"success": True})

@app.route("/api/simulation/start", methods=["POST"])
def api_simulation_start():
    """Start simulation mode."""
    global simulation_running, simulation_thread
    
    if simulation_running:
        return jsonify({"error": "Simulation already running"}), 400
    
    simulation_running = True
    simulation_thread = threading.Thread(target=run_simulation, daemon=True)
    simulation_thread.start()
    
    return jsonify({"success": True})

@app.route("/api/simulation/stop", methods=["POST"])
def api_simulation_stop():
    """Stop simulation mode."""
    global simulation_running
    simulation_running = False
    add_log("⏹️ Simulation stopped")
    return jsonify({"success": True})

@app.route("/api/simulation/status")
def api_simulation_status():
    """Get simulation status."""
    return jsonify({"running": simulation_running})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the game."""
    global board, move_history, game_start_time, simulation_running
    simulation_running = False
    board = chess.Board()
    move_history = []
    game_start_time = datetime.now()
    add_log("🔄 New game started!")
    return jsonify({"fen": board.fen()})

@app.route("/api/stats")
def api_stats():
    """Return game statistics."""
    positions_trained = count_trained_positions()
    gpu_status = "METAL ACTIVE" if device.type == "mps" else device.type.upper()
    
    game_time = "00:00"
    if game_start_time:
        elapsed = datetime.now() - game_start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        game_time = f"{minutes:02d}:{seconds:02d}"
    
    load_model()
    
    return jsonify({
        "positions_trained": positions_trained,
        "gpu_status": gpu_status,
        "model_loaded": model_loaded,
        "model_version": model_version,
        "move_count": len(move_history),
        "game_time": game_time,
        "game_mode": game_mode,
        "player_color": player_color,
        "stockfish_elo": stockfish_elo,
        "simulation_running": simulation_running
    })

@app.route("/api/logs")
def api_logs():
    """Return recent engine logs."""
    return jsonify({"logs": engine_logs})

@app.route("/api/pgn")
def api_pgn():
    """Generate PGN for current game."""
    game = chess.pgn.Game()
    game.headers["Event"] = "BugzyEngine Game"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Player" if player_color == "white" else "BugzyEngine"
    game.headers["Black"] = "BugzyEngine" if player_color == "white" else "Player"
    
    node = game
    temp_board = chess.Board()
    for move_data in move_history:
        move_san = move_data["move"]
        try:
            move = temp_board.parse_san(move_san)
            node = node.add_variation(move)
            temp_board.push(move)
        except:
            pass
    
    return jsonify({"pgn": str(game)})

# --- HTML Template (Truncated for brevity - will be in separate message) ---
CYBERPUNK_HTML_V5 = """
<!DOCTYPE html>
<html>
<head>
    <title>BugzyEngine v5.0</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ffff;
            padding: 10px;
            overflow: hidden;
            height: 100vh;
        }
        .container { 
            max-width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        h1 {
            text-align: center;
            font-size: 2em;
            text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff;
            margin-bottom: 10px;
            letter-spacing: 3px;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 280px 1fr 350px;
            gap: 15px;
            flex: 1;
            overflow: hidden;
        }
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #00ffff;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        h2 {
            color: #ff0055;
            text-shadow: 0 0 10px #ff0055;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        #board {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
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
            width: 100%;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ffff;
        }
        .btn-danger {
            background: linear-gradient(135deg, #ff0055 0%, #ff4488 100%);
        }
        .btn-success {
            background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        }
        select, input[type="range"] {
            width: 100%;
            padding: 10px;
            background: rgba(0, 255, 255, 0.1);
            border: 1px solid #00ffff;
            color: #00ffff;
            border-radius: 5px;
            margin-bottom: 10px;
            font-family: 'Courier New', monospace;
        }
        .color-selector {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .color-btn {
            flex: 1;
            padding: 15px;
            border: 2px solid #00ffff;
            background: rgba(0, 255, 255, 0.1);
            color: #00ffff;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.3s;
        }
        .color-btn.active {
            background: #00ffff;
            color: #000;
            box-shadow: 0 0 20px #00ffff;
        }
        .suggestions-list {
            list-style: none;
        }
        .suggestion-item {
            background: rgba(0, 255, 255, 0.1);
            padding: 10px;
            margin-bottom: 8px;
            border-left: 3px solid #00ffff;
            border-radius: 3px;
        }
        .suggestion-move {
            color: #00ffff;
            font-weight: bold;
            font-size: 1.1em;
        }
        .suggestion-eval {
            color: #ff0055;
            float: right;
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
        .logs {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            padding: 10px;
            flex: 1;
            overflow-y: auto;
            font-size: 0.75em;
            color: #00ff88;
            border-radius: 5px;
        }
        .log-entry {
            margin-bottom: 5px;
            padding: 3px 0;
            border-bottom: 1px solid rgba(0, 255, 136, 0.2);
        }
        .move-history {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #ff0055;
            padding: 10px;
            height: 150px;
            overflow-y: auto;
            border-radius: 5px;
            font-size: 0.85em;
        }
        .game-status {
            text-align: center;
            padding: 10px;
            background: rgba(0, 255, 255, 0.2);
            border: 1px solid #00ffff;
            border-radius: 5px;
            margin-bottom: 10px;
            color: #00ffff;
            font-size: 1.2em;
        }
        .elo-display {
            text-align: center;
            color: #ff0055;
            font-size: 1.2em;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ BUGZYENGINE 5.0 ⚡</h1>
        
        <div class="main-grid">
            <!-- Left Panel: Controls -->
            <div class="panel">
                <h2>🎮 GAME MODE</h2>
                <select id="game-mode" onchange="changeGameMode()">
                    <option value="manual">Manual Play</option>
                    <option value="simulation">Simulation Mode</option>
                </select>
                
                <h2 style="margin-top: 20px;">🎨 YOUR COLOR</h2>
                <div class="color-selector">
                    <div class="color-btn active" id="white-btn" onclick="selectColor('white')">⚪ WHITE</div>
                    <div class="color-btn" id="black-btn" onclick="selectColor('black')">⚫ BLACK</div>
                </div>
                
                <h2>🎯 STOCKFISH ELO</h2>
                <input type="range" id="stockfish-elo" min="500" max="3200" step="500" value="2000" oninput="updateElo()">
                <div class="elo-display" id="elo-display">ELO: 2000</div>
                
                <h2 style="margin-top: 20px;">⚙️ CONTROLS</h2>
                <button class="btn" onclick="resetGame()">🔄 NEW GAME</button>
                <button class="btn btn-success" id="sim-btn" onclick="toggleSimulation()" style="display:none;">▶️ START SIMULATION</button>
                <button class="btn btn-danger" onclick="copyPGN()">📋 COPY PGN</button>
                
                <h2 style="margin-top: 20px;">📊 STATS</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-label">GPU STATUS</div>
                        <div class="stat-value" id="gpu-status">LOADING...</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MODEL VERSION</div>
                        <div class="stat-value" id="model-version">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">MOVE COUNT</div>
                        <div class="stat-value" id="move-count">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">GAME TIME</div>
                        <div class="stat-value" id="game-time">00:00</div>
                    </div>
                </div>
                
                <h2 style="margin-top: 20px;">📜 MOVE HISTORY</h2>
                <div class="move-history" id="move-history"></div>
            </div>
            
            <!-- Center Panel: Board -->
            <div class="panel">
                <div class="game-status" id="game-status">White to move</div>
                <div id="board"></div>
            </div>
            
            <!-- Right Panel: Suggestions & Logs -->
            <div class="panel">
                <h2>💡 MOVE SUGGESTIONS</h2>
                <ul class="suggestions-list" id="suggestions"></ul>
                
                <h2 style="margin-top: 20px;">🔥 ENGINE LOGS</h2>
                <div class="logs" id="logs"></div>
            </div>
        </div>
    </div>
    
    <script>
        let game = new Chess();
        let board = null;
        let gameMode = 'manual';
        let playerColor = 'white';
        let stockfishElo = 2000;
        let simulationRunning = false;
        
        function onDragStart(source, piece, position, orientation) {
            if (game.game_over()) return false;
            if (gameMode === 'simulation') return false;
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
                return false;
            }
            if ((playerColor === 'white' && game.turn() === 'b') ||
                (playerColor === 'black' && game.turn() === 'w')) {
                return false;
            }
        }
        
        function onDrop(source, target) {
            let move = game.move({
                from: source,
                to: target,
                promotion: 'q'
            });
            
            if (move === null) return 'snapback';
            
            updateStatus();
            makeEngineMove(move.from + move.to);
        }
        
        function onSnapEnd() {
            board.position(game.fen());
        }
        
        function makeEngineMove(playerMove) {
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({move: playerMove})
            })
            .then(response => response.json())
            .then(data => {
                if (data.fen) {
                    game.load(data.fen);
                    board.position(data.fen);
                    updateStatus();
                    updateSuggestions(data.suggestions || []);
                }
            });
        }
        
        function updateStatus() {
            let status = '';
            if (game.in_checkmate()) {
                status = '🏁 Checkmate!';
            } else if (game.in_draw()) {
                status = '🤝 Draw';
            } else if (game.in_check()) {
                status = game.turn() === 'w' ? '⚠️ White in check' : '⚠️ Black in check';
            } else {
                status = game.turn() === 'w' ? 'White to move' : 'Black to move';
            }
            document.getElementById('game-status').textContent = status;
        }
        
        function updateSuggestions(suggestions) {
            let html = '';
            suggestions.forEach(sug => {
                html += `<li class="suggestion-item">
                    <span class="suggestion-move">${sug.rank}. ${sug.move}</span>
                    <span class="suggestion-eval">${sug.eval}</span>
                </li>`;
            });
            document.getElementById('suggestions').innerHTML = html || '<li class="suggestion-item">No suggestions available</li>';
        }
        
        function changeGameMode() {
            gameMode = document.getElementById('game-mode').value;
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: gameMode})
            });
            
            if (gameMode === 'simulation') {
                document.getElementById('sim-btn').style.display = 'block';
            } else {
                document.getElementById('sim-btn').style.display = 'none';
            }
        }
        
        function selectColor(color) {
            playerColor = color;
            document.getElementById('white-btn').classList.remove('active');
            document.getElementById('black-btn').classList.remove('active');
            document.getElementById(color + '-btn').classList.add('active');
            
            board.orientation(color === 'white' ? 'white' : 'black');
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({color: color})
            });
        }
        
        function updateElo() {
            stockfishElo = parseInt(document.getElementById('stockfish-elo').value);
            document.getElementById('elo-display').textContent = 'ELO: ' + stockfishElo;
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({stockfish_elo: stockfishElo})
            });
        }
        
        function toggleSimulation() {
            if (!simulationRunning) {
                fetch('/api/simulation/start', {method: 'POST'})
                .then(() => {
                    simulationRunning = true;
                    document.getElementById('sim-btn').textContent = '⏹️ STOP SIMULATION';
                    document.getElementById('sim-btn').classList.add('btn-danger');
                    document.getElementById('sim-btn').classList.remove('btn-success');
                });
            } else {
                fetch('/api/simulation/stop', {method: 'POST'})
                .then(() => {
                    simulationRunning = false;
                    document.getElementById('sim-btn').textContent = '▶️ START SIMULATION';
                    document.getElementById('sim-btn').classList.remove('btn-danger');
                    document.getElementById('sim-btn').classList.add('btn-success');
                });
            }
        }
        
        function resetGame() {
            fetch('/api/reset', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                game.reset();
                board.start();
                updateStatus();
                document.getElementById('suggestions').innerHTML = '';
                document.getElementById('move-history').innerHTML = '';
            });
        }
        
        function copyPGN() {
            fetch('/api/pgn')
            .then(response => response.json())
            .then(data => {
                navigator.clipboard.writeText(data.pgn);
                alert('PGN copied to clipboard!');
            });
        }
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gpu-status').textContent = data.gpu_status;
                document.getElementById('model-version').textContent = data.model_version;
                document.getElementById('move-count').textContent = data.move_count;
                document.getElementById('game-time').textContent = data.game_time;
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
        
        function updateBoard() {
            if (gameMode === 'simulation') {
                fetch('/api/board')
                .then(response => response.json())
                .then(data => {
                    game.load(data.fen);
                    board.position(data.fen);
                    updateStatus();
                    
                    // Update move history
                    if (data.move_history) {
                        let html = '';
                        data.move_history.forEach((m, i) => {
                            html += `<div>${i+1}. ${m.player}: ${m.move}</div>`;
                        });
                        document.getElementById('move-history').innerHTML = html;
                    }
                });
            }
        }
        
        // Initialize
        let config = {
            draggable: true,
            position: 'start',
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
        };
        board = Chessboard('board', config);
        
        updateStatus();
        setInterval(updateStats, 2000);
        setInterval(updateLogs, 2000);
        setInterval(updateBoard, 1000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    add_log("⚡ BugzyEngine 5.0 started!")
    add_log(f"🖥️ GPU Device: {device.type}")
    add_log(f"🧠 Model loaded: {model_loaded}")
    print(f"Starting BugzyEngine v5.0 on http://localhost:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=True, use_reloader=False)
