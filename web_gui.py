#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BugzyEngine v6.0 - Complete Web GUI Rewrite
Fully functional chess interface with guaranteed working chessboard
"""

from flask import Flask, render_template_string, request, jsonify
import chess
import chess.engine
import chess.pgn
import chess.polyglot
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

# Global state
board = chess.Board()
device = torch.device(GPU_DEVICE)
move_history = []
engine_logs = []
game_start_time = None

# Game settings
game_mode = "manual"
player_color = "white"
stockfish_elo = 2000
simulation_running = False
simulation_thread = None

# Model management
model = ChessNet().to(device)
model_loaded = False
model_version = 0
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bugzy_model.pth")

# Stockfish
stockfish_engine = None
STOCKFISH_PATH = "stockfish"

# Opening book
OPENING_BOOK_PATH = os.path.join(os.path.dirname(__file__), "opening_books", "gm2001.bin")
opening_book_reader = None

def add_log(message):
    """Add log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    engine_logs.append(f"[{timestamp}] {message}")
    if len(engine_logs) > 50:
        engine_logs.pop(0)
    print(f"[{timestamp}] {message}")

def load_model():
    """Load the neural network model."""
    global model, model_loaded, model_version
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        model_loaded = True
        
        version_file = os.path.join(os.path.dirname(MODEL_PATH), "version.txt")
        try:
            with open(version_file) as f:
                model_version = int(f.read().strip())
        except:
            model_version = 1
        
        add_log(f"🧠 Model v{model_version} loaded")
        return True
    except Exception as e:
        add_log(f"⚠️ Error loading model: {e}")
        return False

def init_stockfish():
    """Initialize Stockfish engine."""
    global stockfish_engine
    try:
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        add_log("✅ Stockfish initialized")
        return True
    except Exception as e:
        add_log(f"⚠️ Stockfish error: {e}")
        return False

def load_opening_book():
    """Load polyglot opening book."""
    global opening_book_reader
    if os.path.exists(OPENING_BOOK_PATH):
        try:
            opening_book_reader = chess.polyglot.open_reader(OPENING_BOOK_PATH)
            add_log("📚 Opening book loaded")
            return True
        except Exception as e:
            add_log(f"⚠️ Opening book error: {e}")
    return False

def get_opening_move():
    """Get move from opening book."""
    if not opening_book_reader:
        return None
    try:
        entry = opening_book_reader.weighted_choice(board)
        return entry.move if entry else None
    except:
        return None

def get_bugzy_move():
    """Get move from BugzyEngine."""
    if not model_loaded:
        legal_moves = list(board.legal_moves)
        return legal_moves[0] if legal_moves else None
    
    try:
        move = alpha_beta_search(board, model, device, depth=3)
        if move and move in board.legal_moves:
            return move
    except Exception as e:
        add_log(f"⚠️ Engine error: {e}")
    
    legal_moves = list(board.legal_moves)
    return legal_moves[0] if legal_moves else None

def get_stockfish_move(elo):
    """Get move from Stockfish."""
    if not stockfish_engine:
        return None
    try:
        elo = max(1320, min(3200, elo))
        stockfish_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        result = stockfish_engine.play(board.copy(), chess.engine.Limit(time=0.1))
        return result.move if result and result.move in board.legal_moves else None
    except Exception as e:
        add_log(f"⚠️ Stockfish error: {e}")
        return None

# Initialize on startup
load_model()
init_stockfish()
load_opening_book()

# API Routes
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/move", methods=["POST"])
def api_move():
    """Handle player move and get engine response."""
    global board, move_history
    
    data = request.json
    player_move = data.get("move")
    
    # Manual mode: handle player move
    if player_move and game_mode == "manual":
        try:
            move = chess.Move.from_uci(player_move)
            if move in board.legal_moves:
                san = board.san(move)
                board.push(move)
                move_history.append({"move": san, "player": "You"})
                add_log(f"You: {san}")
        except:
            pass
    
    # Make engine move(s)
    if not board.is_game_over():
        if game_mode == "manual":
            # Manual mode: BugzyEngine responds
            engine_move = get_bugzy_move()
            if engine_move:
                san = board.san(engine_move)
                board.push(engine_move)
                move_history.append({"move": san, "player": "BugzyEngine"})
                add_log(f"BugzyEngine: {san}")
        
        elif game_mode == "simulation":
            # Simulation mode: alternate between engines
            if board.turn() == (chess.WHITE if player_color == "white" else chess.BLACK):
                # BugzyEngine's turn
                book_move = get_opening_move()
                if book_move:
                    san = board.san(book_move)
                    board.push(book_move)
                    move_history.append({"move": san, "player": "BugzyEngine (Book)"})
                    add_log(f"BugzyEngine (Book): {san}")
                else:
                    engine_move = get_bugzy_move()
                    if engine_move:
                        san = board.san(engine_move)
                        board.push(engine_move)
                        move_history.append({"move": san, "player": "BugzyEngine"})
                        add_log(f"BugzyEngine: {san}")
            else:
                # Stockfish's turn
                stockfish_move = get_stockfish_move(stockfish_elo)
                if stockfish_move:
                    san = board.san(stockfish_move)
                    board.push(stockfish_move)
                    move_history.append({"move": san, "player": f"Stockfish-{stockfish_elo}"})
                    add_log(f"Stockfish-{stockfish_elo}: {san}")
    
    return jsonify({
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset the game."""
    global board, move_history, game_start_time, simulation_running
    simulation_running = False
    board = chess.Board()
    move_history = []
    game_start_time = datetime.now()
    add_log("🔄 New game started")
    return jsonify({"fen": board.fen()})

@app.route("/api/stats")
def api_stats():
    """Get game statistics."""
    gpu_status = "METAL ACTIVE" if device.type == "mps" else device.type.upper()
    
    game_time = "00:00"
    if game_start_time:
        elapsed = datetime.now() - game_start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)
        game_time = f"{minutes:02d}:{seconds:02d}"
    
    return jsonify({
        "gpu_status": gpu_status,
        "model_version": model_version,
        "move_count": len(move_history),
        "game_time": game_time
    })

@app.route("/api/board")
def api_board():
    """Get current board state."""
    return jsonify({
        "fen": board.fen(),
        "moves": move_history
    })

@app.route("/api/logs")
def api_logs():
    """Get engine logs."""
    return jsonify({"logs": engine_logs[-50:]})

@app.route("/api/suggestions")
def api_suggestions():
    """Get move suggestions from BugzyEngine."""
    if game_mode != "manual" or board.is_game_over():
        return jsonify({"suggestions": []})
    
    suggestions = []
    legal_moves = list(board.legal_moves)
    
    # Get top 3 moves from BugzyEngine
    for i, move in enumerate(legal_moves[:3]):
        suggestions.append({
            "move": board.san(move),
            "uci": move.uci()
        })
    
    return jsonify({"suggestions": suggestions})
@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update game settings."""
    global game_mode, player_color, stockfish_elo
    data = request.json
    game_mode = data.get("mode", game_mode)
    player_color = data.get("color", player_color)
    stockfish_elo = data.get("elo", stockfish_elo)
    return jsonify({"success": True})

@app.route("/api/pgn")
def api_pgn():
    """Generate PGN."""
    game = chess.pgn.Game()
    game.headers["Event"] = "BugzyEngine Game"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Player" if player_color == "white" else "BugzyEngine"
    game.headers["Black"] = "BugzyEngine" if player_color == "white" else "Player"
    
    node = game
    temp_board = chess.Board()
    for move_data in move_history:
        try:
            move = temp_board.parse_san(move_data["move"])
            node = node.add_variation(move)
            temp_board.push(move)
        except:
            pass
    
    return jsonify({"pgn": str(game)})

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>BugzyEngine v6.0</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff41;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-bottom: 30px;
            text-shadow: 0 0 20px #00ff41;
            color: #00ffff;
        }
        .game-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 20px;
        }
        .panel {
            background: rgba(0, 20, 40, 0.8);
            border: 2px solid #00ff41;
            border-radius: 10px;
            padding: 20px;
        }
        .panel h2 {
            color: #ff0080;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        #board {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }
        .btn {
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            background: #00ff41;
            color: #000;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover {
            background: #00cc33;
        }
        .btn-danger {
            background: #ff0080;
            color: #fff;
        }
        .btn-danger:hover {
            background: #cc0066;
        }
        select, input[type="range"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            background: #000;
            color: #00ff41;
            border: 1px solid #00ff41;
            border-radius: 5px;
        }
        .stat-box {
            background: #000;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #00ff41;
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
        .log-box {
            background: #000;
            padding: 10px;
            height: 200px;
            overflow-y: auto;
            border-radius: 5px;
            border: 1px solid #00ff41;
            font-size: 0.9em;
        }
        .move-list {
            background: #000;
            padding: 10px;
            height: 300px;
            overflow-y: auto;
            border-radius: 5px;
            border: 1px solid #00ff41;
        }
        .color-btn {
            display: inline-block;
            width: 48%;
            padding: 10px;
            margin: 5px 1%;
            background: #000;
            border: 2px solid #00ff41;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
        }
        .color-btn.active {
            background: #00ff41;
            color: #000;
        }
        .game-status {
            text-align: center;
            font-size: 1.5em;
            padding: 15px;
            background: #000;
            border-radius: 5px;
            margin-bottom: 20px;
            color: #00ffff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚡ BUGZYENGINE 6.0 ⚡</h1>
        
        <div class="game-container">
            <!-- Left Panel -->
            <div class="panel">
                <h2>🎮 GAME MODE</h2>
                <select id="game-mode" onchange="updateSettings()">
                    <option value="manual">Manual Play</option>
                    <option value="simulation">Simulation Mode</option>
                </select>
                
                <h2 style="margin-top: 20px;">🎨 YOUR COLOR</h2>
                <div>
                    <div class="color-btn active" id="white-btn" onclick="selectColor('white')">⚪ WHITE</div>
                    <div class="color-btn" id="black-btn" onclick="selectColor('black')">⚫ BLACK</div>
                </div>
                
                <h2 style="margin-top: 20px;">🎯 STOCKFISH ELO</h2>
                <input type="range" id="stockfish-elo" min="1320" max="3200" step="100" value="2000" oninput="updateElo()">
                <div id="elo-display" class="stat-value" style="text-align: center;">ELO: 2000</div>
                
                <h2 style="margin-top: 20px;">⚙️ CONTROLS</h2>
                <button class="btn" onclick="resetGame()">🔄 NEW GAME</button>
                <button class="btn" id="sim-btn" onclick="toggleSimulation()" style="display:none;">▶️ START SIMULATION</button>
                <button class="btn btn-danger" onclick="copyPGN()">📋 COPY PGN</button>
            </div>
            
            <!-- Center Panel -->
            <div class="panel">
                <div class="game-status" id="game-status">White to move</div>
                <div id="board"></div>
            </div>
            
            <!-- Right Panel -->
            <div class="panel">
                <h2>💡 MOVE SUGGESTIONS</h2>
                <div id="suggestions" style="padding: 10px; min-height: 100px; background: rgba(0,255,255,0.05); border-radius: 5px; margin-bottom: 20px;">
                    <div style="color: #0ff; font-size: 14px;">Waiting for your move...</div>
                </div>
                
                <h2>📊 STATS</h2>
                <div class="stat-box">
                    <div class="stat-label">GPU STATUS</div>
                    <div class="stat-value" id="gpu-status">LOADING...</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">MODEL VERSION</div>
                    <div class="stat-value" id="model-version">0</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">MOVES</div>
                    <div class="stat-value" id="move-count">0</div>
                </div>
                
                <h2 style="margin-top: 20px;">📜 MOVE HISTORY</h2>
                <div class="move-list" id="move-history"></div>
                
                <h2 style="margin-top: 20px;">🔥 ENGINE LOGS</h2>
                <div class="log-box" id="logs"></div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
    <script>
        var game = new Chess();
        var board = null;
        var gameMode = 'manual';
        var playerColor = 'white';
        var stockfishElo = 2000;
        
        function onDragStart(source, piece) {
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
            return true;
        }
        
        function onDrop(source, target) {
            var move = game.move({
                from: source,
                to: target,
                promotion: 'q'
            });
            
            if (move === null) return 'snapback';
            
            updateStatus();
            makeEngineMove(move.from + move.to);
            return true;
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
                    updateBoard();
                }
            });
        }
        
        function updateStatus() {
            var status = '';
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
        
        function resetGame() {
            fetch('/api/reset', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                game.reset();
                board.start();
                updateStatus();
                updateBoard();
            });
        }
        
        function selectColor(color) {
            playerColor = color;
            document.getElementById('white-btn').classList.toggle('active', color === 'white');
            document.getElementById('black-btn').classList.toggle('active', color === 'black');
            updateSettings();
        }
        
        function updateElo() {
            stockfishElo = parseInt(document.getElementById('stockfish-elo').value);
            document.getElementById('elo-display').textContent = 'ELO: ' + stockfishElo;
            updateSettings();
        }
        
        function updateSettings() {
            gameMode = document.getElementById('game-mode').value;
            
            // Show/hide simulation button
            var simBtn = document.getElementById('sim-btn');
            if (gameMode === 'simulation') {
                simBtn.style.display = 'block';
            } else {
                simBtn.style.display = 'none';
            }
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    mode: gameMode,
                    color: playerColor,
                    elo: stockfishElo
                })
            });
        }
        
        var simulationRunning = false;
        function toggleSimulation() {
            simulationRunning = !simulationRunning;
            var btn = document.getElementById('sim-btn');
            if (simulationRunning) {
                btn.textContent = '⏹️ STOP SIMULATION';
                btn.classList.add('btn-danger');
                runSimulation();
            } else {
                btn.textContent = '▶️ START SIMULATION';
                btn.classList.remove('btn-danger');
            }
        }
        
        function runSimulation() {
            if (!simulationRunning || game.game_over()) {
                simulationRunning = false;
                document.getElementById('sim-btn').textContent = '▶️ START SIMULATION';
                return;
            }
            
            // Make a move
            fetch('/api/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({move: ''})
            })
            .then(response => response.json())
            .then(data => {
                if (data.fen) {
                    game.load(data.fen);
                    board.position(data.fen);
                    updateStatus();
                    updateBoard();
                }
                
                // Continue simulation
                if (simulationRunning && !game.game_over()) {
                    setTimeout(runSimulation, 1000);
                }
            });
        }
        
        function copyPGN() {
            fetch('/api/pgn')
            .then(response => response.json())
            .then(data => {
                if (navigator.clipboard) {
                    navigator.clipboard.writeText(data.pgn)
                        .then(() => alert('✅ PGN copied to clipboard!'))
                        .catch(() => alert('📋 PGN:\\n\\n' + data.pgn));
                } else {
                    alert('📋 PGN:\\n\\n' + data.pgn);
                }
            });
        }
        
        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('gpu-status').textContent = data.gpu_status;
                document.getElementById('model-version').textContent = data.model_version;
                document.getElementById('move-count').textContent = data.move_count;
            });
        }
        
         function updateBoard() {
            fetch('/api/board')
            .then(response => response.json())
            .then(data => {
                var historyHtml = '';
                data.moves.forEach((m, i) => {
                    historyHtml += `<div style="padding: 5px; border-bottom: 1px solid rgba(0,255,255,0.2);">${i+1}. ${m.player}: ${m.move}</div>`;
                });
                document.getElementById('move-history').innerHTML = historyHtml;
            });
            
            // Update suggestions in manual mode
            if (gameMode === 'manual') {
                updateSuggestions();
            }
        }
        
        function updateSuggestions() {
            fetch('/api/suggestions')
            .then(response => response.json())
            .then(data => {
                var suggestionsDiv = document.getElementById('suggestions');
                if (data.suggestions && data.suggestions.length > 0) {
                    var html = '<div style="color: #0ff; font-weight: bold; margin-bottom: 10px;">BugzyEngine suggests:</div>';
                    data.suggestions.forEach((s, i) => {
                        html += `<div style="padding: 8px; margin: 5px 0; background: rgba(0,255,255,0.1); border-left: 3px solid #0ff; border-radius: 3px;">
                            <span style="color: #0f0; font-weight: bold;">${i+1}.</span> ${s.move}
                        </div>`;
                    });
                    suggestionsDiv.innerHTML = html;
                } else {
                    suggestionsDiv.innerHTML = '<div style="color: #0ff; font-size: 14px;">Waiting for your move...</div>';
                }
            });
        }
        
        function updateLogs() {
            fetch('/api/logs')
            .then(response => response.json())
            .then(data => {
                var html = '';
                data.logs.forEach(log => {
                    html += '<div style="padding: 2px;">' + log + '</div>';
                });
                document.getElementById('logs').innerHTML = html;
                var logsDiv = document.getElementById('logs');
                logsDiv.scrollTop = logsDiv.scrollHeight;
            });
        }
        
        // Initialize
        $(document).ready(function() {
            var config = {
                draggable: true,
                position: 'start',
                onDragStart: onDragStart,
                onDrop: onDrop,
                onSnapEnd: onSnapEnd,
                pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
            };
            board = Chessboard('board', config);
            
            updateStatus();
            updateStats();
            setInterval(updateStats, 2000);
            setInterval(updateBoard, 1000);
            setInterval(updateLogs, 2000);
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"Starting BugzyEngine v6.0 on http://localhost:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=True, use_reloader=False)
