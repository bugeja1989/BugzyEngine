# BugzyEngine Quick Start Guide

Get BugzyEngine up and running on your Mac in 5 minutes!

## Step 1: Install Dependencies

```bash
cd BugzyEngine
chmod +x install.sh
./install.sh
```

This will:
- Install Python 3.11 via Homebrew
- Create a virtual environment
- Install all required packages (PyTorch, Chess, Flask, etc.)
- Verify GPU (MPS) availability

## Step 2: Start BugzyEngine

```bash
chmod +x run.sh
./run.sh
```

This will start three processes:
1. **Data Collector**: Fetches games from Chess.com
2. **Neural Network Trainer**: Trains the model on new games
3. **Web GUI**: Launches on `http://localhost:9443`

## Step 3: Play Chess!

Open your browser to `http://localhost:9443` and start playing!

### How to Make Moves

- Use standard algebraic notation: `e2e4`, `g1f3`, etc.
- The engine will respond with its move
- Click "New Game" to reset the board

## Troubleshooting

### "Python 3.11 not found"
Make sure Homebrew is installed and run:
```bash
brew install python@3.11
```

### "MPS not available"
If you're on Intel Mac, MPS won't be available. The engine will use CPU instead (slower).

### "Port 9443 already in use"
Edit `config.py` and change `WEB_PORT` to another port (e.g., 8080).

### "No model found" warning
This is normal on first run. The engine will play randomly until training data is collected. Games from Chess.com are being downloaded in the background.

## Next Steps

1. **Monitor Training**: Check the console output to see training progress
2. **Add Custom Games**: Drop PGN files into `data/raw_pgn/` for the trainer to process
3. **Tune Configuration**: Edit `config.py` to adjust hyperparameters
4. **Play Tournaments**: Use the engine in UCI format with chess GUIs

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    BugzyEngine System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  Data Collector  │  │ Neural Trainer   │  │  Web GUI   │ │
│  │  (Process 1)     │  │  (Process 2)     │  │ (Process 3)│ │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬───┘ │
│           │                     │                      │     │
│           ├─────────────────────┼──────────────────────┤     │
│           ▼                     ▼                      ▼     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Shared Chess Logic (engine_utils.py)           │ │
│  │  - board_to_tensor()                                    │ │
│  │  - evaluate_position()                                  │ │
│  │  - alpha_beta_search()                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│           │                     │                      │     │
│           ▼                     ▼                      ▼     │
│  ┌──────────────┐    ┌──────────────────┐   ┌──────────────┐│
│  │ Chess.com    │    │  GPU (MPS/CUDA)  │   │ PyTorch      ││
│  │ API          │    │  Acceleration    │   │ Model        ││
│  └──────────────┘    └──────────────────┘   └──────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

✅ **GPU Acceleration**: Full support for Apple Silicon (M-series) GPUs  
✅ **Dynamic Player Discovery**: Automatically finds 500+ Super GMs  
✅ **Continuous Training**: Learns from new games automatically  
✅ **Full Alpha-Beta Search**: Depth 3+ with move ordering  
✅ **Web Interface**: Play via browser on Port 9443  
✅ **Modular Architecture**: Shared logic between trainer and GUI  

## Performance

On Apple M4 Pro with GPU:
- Training: ~1000 positions/second
- Inference: ~100 positions/second
- Model size: ~2 MB

## Support

For issues or questions, check the README.md or open an issue on the project repository.

---

**Enjoy BugzyEngine! ♟️**
