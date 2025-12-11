# BugzyEngine Architecture

This document provides a detailed overview of the BugzyEngine system architecture, design decisions, and component interactions.

## System Overview

BugzyEngine is a distributed chess engine system designed to run on macOS with GPU acceleration. It consists of three independent processes that communicate through shared files and a centralized configuration:

```
┌─────────────────────────────────────────────────────────────────┐
│                      BugzyEngine System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Shared Configuration (config.py)            │   │
│  │  - Web Port (9443)                                       │   │
│  │  - GPU Device (MPS/CUDA/CPU)                             │   │
│  │  - Historical Legends                                    │   │
│  │  - Chess.com API Endpoint                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│     ┌───────────────────┼───────────────────┐                   │
│     │                   │                   │                   │
│     ▼                   ▼                   ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Process 1   │  │  Process 2   │  │  Process 3   │          │
│  │  Collector   │  │  Trainer     │  │  Web GUI     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│     │                   │                   │                   │
│     └───────────────────┼───────────────────┘                   │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Shared Chess Logic (engine_utils.py)             │   │
│  │  - board_to_tensor()    [Convert board to neural input]  │   │
│  │  - evaluate_position()  [Neural network evaluation]      │   │
│  │  - alpha_beta_search()  [Search with move ordering]      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                         │                                        │
│     ┌───────────────────┼───────────────────┐                   │
│     │                   │                   │                   │
│     ▼                   ▼                   ▼                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Chess.com    │  │ GPU (MPS)    │  │ PyTorch      │          │
│  │ API          │  │ Acceleration │  │ Model        │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Collector (process1_chesscom_collector.py)

**Purpose**: Discovers and downloads chess games from the Chess.com API and monitors local PGN files.

**Key Functions**:
- `get_leaderboard_players()`: Fetches top 50 players from leaderboards
- `get_titled_players(title)`: Fetches all players with a given title (GM, IM, etc.)
- `get_player_games(player_username)`: Downloads all games for a specific player
- `is_game_style_match(game, player)`: Filters games by style (wins, sacrifices, short games)

**Data Flow**:
```
Chess.com API
    ↓
Dynamic Player Discovery
    ├─ Top 100 Leaderboard
    ├─ All GMs (ELO > 2600)
    ├─ All IMs (ELO > 2600)
    └─ Historical Legends
    ↓
Game Download & Filtering
    ├─ Wins only
    ├─ Sacrifices
    └─ Short games (< 40 moves)
    ↓
data/raw_pgn/ (PGN files)
```

**Rate Limiting**: 1 request per second to avoid Chess.com API throttling

**Folder Watcher**: Monitors `data/raw_pgn/` for locally-added PGN files (e.g., from Lichess, TWIC)

### 2. Neural Network Trainer (process2_trainer.py)

**Purpose**: Continuously trains a neural network on chess game data using GPU acceleration.

**Architecture**:
```
ChessNet (PyTorch Model)
├─ Input: 12×8×8 tensor (6 piece types × 2 colors × 64 squares)
├─ Conv2D(12→64, 3×3) + ReLU
├─ Conv2D(64→64, 3×3) + ReLU
├─ Flatten → 4096 neurons
├─ Dense(4096→256) + ReLU
└─ Dense(256→1) + Tanh (output: -1 to +1)
```

**Training Pipeline**:
1. Scan `data/raw_pgn/` for new PGN files
2. Parse games and extract positions
3. Convert positions to tensors using `board_to_tensor()`
4. Train on GPU with:
   - Optimizer: Adam (lr=0.001)
   - Loss: Mean Squared Error
   - Epochs: 6 (user preference)
   - Batch Size: 256
5. Atomic save to `models/bugzy_model.pth`
6. Move processed files to `data/processed/`

**GPU Acceleration**:
- Automatically detects and uses Apple Silicon MPS
- Falls back to CUDA on NVIDIA GPUs
- CPU-only mode available as fallback

### 3. Web GUI (web_gui.py)

**Purpose**: Provides an interactive web interface for playing chess against BugzyEngine.

**Routes**:
- `GET /`: Display current board state
- `GET /move?move=<uci>`: Process user move and generate engine response
- `GET /reset`: Reset the board for a new game

**Move Generation**:
1. Parse user's move in UCI format (e.g., `e2e4`)
2. Validate move legality
3. Apply move to board
4. Call `alpha_beta_search(board, depth=3, model=model)`
5. Apply engine's move
6. Return updated board state

**UI Features**:
- SVG chess board visualization
- Auto-refresh every 5 seconds
- Game over detection
- New game button

## Shared Chess Logic (engine_utils.py)

This module contains the core chess logic used by both the trainer and GUI.

### board_to_tensor(board)

Converts a chess.Board object to a PyTorch tensor:

```
Input: chess.Board object
  ↓
Process:
  - Iterate through all 64 squares
  - For each piece, determine type (1-6) and color (0-1)
  - Create 12-channel tensor (6 types × 2 colors)
  - One-hot encode piece positions
  ↓
Output: torch.Tensor of shape (1, 12, 8, 8)
```

### evaluate_position(board, model)

Evaluates a position using the neural network:

```
Input: chess.Board, PyTorch model
  ↓
Process:
  - Convert board to tensor
  - Pass through model (no gradient)
  - Return scalar value (-1 to +1)
  ↓
Output: float (position evaluation)
```

### alpha_beta_search(board, depth, model)

Performs alpha-beta search with neural network evaluation:

```
Algorithm:
  - Recursive minimax with alpha-beta pruning
  - Base case: depth=0 or game over → evaluate with model
  - Move ordering: Try captures first (heuristic)
  - Repetition detection: Avoid repeated positions
  - Return: (best_score, best_move)
```

## Data Flow

### Training Data Pipeline

```
Chess.com API
    ↓
process1_chesscom_collector.py
    ├─ Discover players
    ├─ Download games
    └─ Filter by style
    ↓
data/raw_pgn/ (unprocessed PGN files)
    ↓
process2_trainer.py
    ├─ Parse games
    ├─ Extract positions
    ├─ Convert to tensors
    └─ Train neural network
    ↓
models/bugzy_model.pth (trained model)
    ↓
web_gui.py (inference)
    └─ Play against user
```

### Game Playing Pipeline

```
User Input (UCI move)
    ↓
web_gui.py
    ├─ Validate move
    ├─ Apply to board
    └─ Call alpha_beta_search()
    ↓
engine_utils.py
    ├─ Generate candidate moves
    ├─ Evaluate positions with model
    └─ Return best move
    ↓
web_gui.py
    ├─ Apply engine move
    └─ Display board
    ↓
Browser (SVG visualization)
```

## Configuration

All configuration is centralized in `config.py`:

```python
WEB_PORT = 9443                    # Web GUI port
GPU_DEVICE = "mps"                # GPU device type
HISTORICAL_LEGENDS = [...]         # Players to always include
CHESS_COM_API_BASE_URL = "..."    # API endpoint
```

## File Structure

```
BugzyEngine/
├── config.py                      # Centralized configuration
├── process1_chesscom_collector.py # Data collection process
├── process2_trainer.py            # Training process
├── web_gui.py                     # Web interface
├── neural_network/
│   └── src/
│       └── engine_utils.py        # Shared chess logic
├── data/
│   ├── raw_pgn/                   # Downloaded PGN files
│   └── processed/                 # Processed training data
├── models/                        # Trained model files
├── logs/                          # Training logs
├── requirements.txt               # Python dependencies
├── install.sh                     # Installation script
├── run.sh                         # Launcher script
├── README.md                      # Main documentation
├── QUICKSTART.md                  # Quick start guide
└── ARCHITECTURE.md                # This file
```

## Design Decisions

### Why Three Separate Processes?

1. **Isolation**: Each process can fail independently without crashing the system
2. **Scalability**: Can run on different machines (future enhancement)
3. **Responsiveness**: GUI remains responsive even during training
4. **Modularity**: Easy to replace or upgrade individual components

### Why PyTorch Instead of TensorFlow?

1. **MPS Support**: Better Metal Performance Shaders integration on macOS
2. **Flexibility**: Easier to implement custom training loops
3. **Performance**: Generally faster on Apple Silicon
4. **Community**: Stronger community support for game AI

### Why Atomic Model Saving?

1. **Safety**: Prevents GUI from loading corrupted models
2. **Consistency**: Ensures model file is always in a valid state
3. **Recovery**: Can rollback to previous model if needed

### Why Shared engine_utils.py?

1. **Consistency**: Both trainer and GUI use identical chess logic
2. **Maintainability**: Bug fixes apply to both components
3. **Performance**: Optimizations benefit entire system
4. **Testing**: Easier to test core logic in isolation

## Performance Characteristics

### Training
- **Speed**: ~1000 positions/second on Apple M4 Pro with GPU
- **Memory**: ~4GB GPU VRAM for batch size 256
- **Throughput**: Can process 1M positions in ~1000 seconds

### Inference
- **Speed**: ~100 positions/second on Apple M4 Pro with GPU
- **Latency**: ~10ms per position evaluation
- **Throughput**: Can evaluate 100 positions in ~1 second

### Data Collection
- **Rate**: ~50 games/minute from Chess.com API
- **Bandwidth**: ~1MB/minute
- **Storage**: ~1KB per game (PGN format)

## Future Enhancements

1. **Opening Book**: Dedicated opening book for first 10 moves
2. **Endgame Tablebase**: Perfect play in endgames
3. **Multi-GPU Training**: Distributed training across multiple GPUs
4. **Distributed Collection**: Parallel data collection from multiple sources
5. **Web Dashboard**: Real-time training monitoring
6. **UCI Protocol**: Standard chess engine protocol support
7. **Lichess Integration**: Collect games from Lichess as well
8. **TWIC Database**: Support for The Week in Chess databases

## Deployment Considerations

### Minimum Requirements
- macOS 11+
- Python 3.11+
- 4GB RAM
- 2GB disk space

### Recommended Setup
- macOS 12+ with Apple Silicon (M-series)
- 8GB+ RAM
- 10GB+ disk space
- Stable internet connection

### Production Deployment
- Run on dedicated Mac mini or Mac Studio
- Use systemd or launchd for process management
- Monitor training progress with logging
- Backup models regularly
- Use external storage for large PGN collections

## Security Considerations

1. **API Rate Limiting**: Respects Chess.com API limits
2. **Local Storage**: All data stored locally, no cloud uploads
3. **No Authentication**: Web GUI has no authentication (local use only)
4. **Sandboxing**: Each process runs independently

---

**Architecture designed for extensibility and performance on macOS with GPU acceleration.**
