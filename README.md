# ⚡ BugzyEngine 3.0 ⚡

**The Cyberpunk Chess Engine That Beats Stockfish Through Chaos**

BugzyEngine is a neural chess engine designed to beat Stockfish by playing dangerous, chaotic, and psychological chess. Built with PyTorch, trained on 500+ Super GMs, and featuring a stunning Cyberpunk web interface.

---

## 🎯 Project Goal

**Beat Stockfish** by learning from the most aggressive, sacrificial, and complex games ever played. BugzyEngine doesn't play "perfect" chess—it plays **dangerous** chess.

---

## ✨ Features

### 🧠 Neural Network
- **PyTorch-based CNN** (12→64→64→256→1 architecture)
- **Full GPU acceleration** via Apple Metal Performance Shaders (MPS)
- **Continuous training** on new game data (6 epochs, batch size 256)
- **Atomic model saving** to prevent corruption

### 📥 Data Collector
- **Dynamic GM discovery**: Automatically finds 500+ Super GMs via Chess.com API
- **Async/parallel downloads**: 20 players processed simultaneously
- **Advanced filtering**:
  - Wins only (or draws with >95% accuracy)
  - Sacrifice detection (material balance tracking)
  - Complexity scoring (captures + checks)
- **Folder watcher**: Drop PGN files from TWIC or Lichess
- **Database tracking**: Never downloads the same game twice

### 🎮 Cyberpunk Web GUI
- **Drag-and-drop chess interface** using chessboard.js
- **Real-time stats dashboard**:
  - GPU Status (METAL ACTIVE)
  - Positions Trained
  - Model Status
  - Move Count
- **Engine logs** with timestamps
- **Move history** panel
- **PGN export** functionality
- **Cyberpunk aesthetic**: Black/Neon Blue/Pink/Green

### 🔧 Architecture
- **Unified chess logic** in `engine_utils.py`
- **Alpha-beta search** with move ordering
- **Centralized configuration** (`config.py`)
- **Comprehensive logging** system

---

## 🚀 Quick Start

### Prerequisites
- **macOS** (Apple Silicon recommended for MPS support)
- **Python 3.11+**
- **Homebrew** (for Python installation)

### Installation

```bash
# Clone the repository
git clone git@github.com:bugeja1989/BugzyEngine.git
cd BugzyEngine

# Run the installer
chmod +x install.sh
./install.sh
```

### Running BugzyEngine

```bash
# Start all processes in background
./run.sh

# Check status
./status.sh

# Stop all processes
./stop.sh
```

### Access the GUI

Open your browser and navigate to:
```
http://localhost:9443
```

---

## 📊 System Architecture

BugzyEngine consists of three processes:

### 1. Data Collector (`process1_chesscom_collector.py`)
- Discovers and downloads games from 500+ Super GMs
- Filters for attacking, sacrificial, complex games
- Runs continuously, checking for new games every hour

### 2. Neural Network Trainer (`process2_trainer.py`)
- Watches for new PGN files
- Trains the neural network on GPU
- Saves models atomically
- Runs continuously in background

### 3. Web GUI (`web_gui.py`)
- Interactive chess interface on Port 9443
- Real-time stats and logs
- Drag-and-drop piece movement
- Full alpha-beta search integration

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# System
WEB_PORT = 9443
GPU_DEVICE = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" for CPU-only

# Collector
COLLECTOR_CONCURRENCY = 20  # Players to process in parallel
ELO_THRESHOLD = 2600  # Minimum ELO for titled players
TOP_N_PLAYERS = 100  # Number of top players from leaderboards

# Filtering
MIN_COMPLEXITY_SCORE = 15  # Minimum captures + checks
MIN_SACRIFICES = 1  # Minimum sacrifices required
MIN_DRAW_COMPLEXITY = 25  # Complexity threshold for draws

# Trainer
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 6

# Search
SEARCH_DEPTH = 3
```

---

## 📁 Project Structure

```
BugzyEngine/
├── config.py                          # Centralized configuration
├── logging_config.py                  # Logging setup
├── process1_chesscom_collector.py     # Data collector
├── process2_trainer.py                # Neural trainer
├── web_gui.py                         # Cyberpunk web interface
├── neural_network/src/engine_utils.py # Shared chess logic
├── data/
│   ├── raw_pgn/                       # Downloaded games
│   ├── processed/                     # Processed games
│   └── collector.db                   # SQLite tracking database
├── models/                            # Trained models
├── logs/                              # Log files
├── install.sh                         # Installation script
├── run.sh                             # Launcher (background mode)
├── stop.sh                            # Shutdown script
├── status.sh                          # Status checker
└── README.md                          # This file
```

---

## 🎨 Cyberpunk GUI Features

### Control Panel
- **NEW GAME**: Reset the board
- **COPY PGN**: Export game to clipboard

### Stats Dashboard
- **GPU STATUS**: Shows METAL ACTIVE (or CUDA/CPU)
- **POSITIONS TRAINED**: Total positions learned
- **MODEL STATUS**: LOADED or RANDOM
- **MOVE COUNT**: Moves in current game

### Engine Logs
Real-time scrolling logs showing:
- Player moves
- Engine moves with [CHAOS MODE] indicator
- Thinking process
- Game over notifications

### Move History
Scrollable list of all moves in the current game

---

## 🔥 Advanced Usage

### Monitor Logs

```bash
# Watch all logs in real-time
tail -f logs/*.log

# Watch specific process
tail -f logs/collector.log
tail -f logs/trainer.log
```

### Manual Data Collection

Drop PGN files into `data/raw_pgn/` and the collector will process them automatically.

### Training Status

Check `logs/trainer.log` to see:
- Number of positions being trained
- Loss per epoch
- Model save confirmations

---

## 🏆 Training Methodology

BugzyEngine follows a four-phase, quality-first approach:

1. **Phase 1**: Top 100 Super GMs (500 games each)
2. **Phase 2**: Historical legends (Tal, Fischer, Kasparov, etc.)
3. **Phase 3**: Top 1000 players (500 games each)
4. **Phase 4**: Continuous retraining (+1000 games per cycle)

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 9443
lsof -ti:9443 | xargs kill -9
```

### GPU Not Detected
Check `logs/trainer.log` for device information. If MPS is not available, the engine will fall back to CPU.

### No Games Collected
- Check `logs/collector.log` for API errors
- Verify internet connection
- Ensure Chess.com API is accessible

### Model Not Loading
- Check if `models/bugzy_model.pth` exists
- Verify file permissions
- Check `logs/webgui_stdout.log` for errors

---

## 📝 Development

### Running Individual Processes

```bash
# Activate virtual environment
source venv/bin/activate

# Run collector only
python3 process1_chesscom_collector.py

# Run trainer only
python3 process2_trainer.py

# Run web GUI only
python3 web_gui.py
```

### Testing

```bash
# Check for syntax errors
python3.11 -m py_compile *.py

# Run with verbose output
python3 web_gui.py
```

---

## 🤝 Contributing

BugzyEngine is designed for collaboration with Chess.com and the chess AI community. Contributions welcome!

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🔗 Links

- **Repository**: https://github.com/bugeja1989/BugzyEngine
- **SSH Clone**: `git clone git@github.com:bugeja1989/BugzyEngine.git`

---

## 🎯 Roadmap

### Completed ✅
- Dynamic GM discovery (500+ players)
- Async/parallel data collection
- Advanced game filtering (sacrifices, complexity)
- Cyberpunk web GUI with drag-and-drop
- Real-time stats dashboard
- Centralized logging
- Background process management

### In Progress 🚧
- Opening book integration (first 10 moves)
- Lichess API integration
- Enhanced sacrifice detection

### Planned 🔮
- Stockfish simulation mode
- UCI protocol support
- Multi-GPU training
- Real-time training dashboard
- Mobile-responsive GUI

---

**Built with ❤️ and ⚡ by Manus AI**

*"Don't play perfect chess. Play dangerous chess."*
