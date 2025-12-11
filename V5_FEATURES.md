# BugzyEngine v5.0 - Complete Game Mode System

## 🎮 Game Modes

### 1. Manual Mode (Default)
**You vs BugzyEngine**

- **Choose your color**: White or Black
- **Drag and drop pieces**: Natural gameplay
- **Engine responds automatically**: BugzyEngine makes counter-moves
- **Real-time move suggestions**: See top 3 moves with evaluations
- **Learn from the engine**: Understand what the AI recommends

**How to use:**
1. Select "Manual Play" from Game Mode dropdown
2. Choose your color (White/Black)
3. Make your move by dragging pieces
4. BugzyEngine responds automatically
5. Check "Move Suggestions" panel for best moves

### 2. Simulation Mode
**BugzyEngine vs Stockfish (Auto-play)**

- **Watch the battle**: Both sides move automatically
- **Select Stockfish ELO**: 500, 1000, 1500, 2000, 2500, 3000, 3200+
- **Pause/Resume**: Control simulation with Start/Stop button
- **Real-time logs**: See every move as it happens
- **Statistics**: Track game progress

**How to use:**
1. Select "Simulation Mode" from Game Mode dropdown
2. Adjust Stockfish ELO slider (500-3200)
3. Click "START SIMULATION"
4. Watch BugzyEngine battle Stockfish!
5. Click "STOP SIMULATION" to pause

## 🎯 Stockfish ELO Levels

| ELO | Skill Level | Description |
|-----|-------------|-------------|
| **500** | Absolute Beginner | Makes frequent blunders, easy to beat |
| **1000** | Novice | Basic understanding, still makes mistakes |
| **1500** | Intermediate | Solid fundamentals, competitive play |
| **2000** | Advanced | Strong tactical awareness |
| **2500** | Master | Expert-level play, very challenging |
| **3000** | Super GM | World-class strength |
| **3200+** | Maximum | Superhuman, nearly unbeatable |

**Perfect for:**
- Testing BugzyEngine at different skill levels
- Training against progressively harder opponents
- Benchmarking your AI's improvement

## 💡 Move Suggestions

The suggestions panel shows:
- **Rank**: 1st, 2nd, 3rd best move
- **Move**: Standard Algebraic Notation (e.g., "Nf3", "Qxd5")
- **Evaluation**: Centipawn score or mate in X

**Example:**
```
1. Nf3    +0.35
2. e4     +0.28
3. d4     +0.15
```

**Positive score** = Good for current player  
**Negative score** = Bad for current player  
**M5** = Mate in 5 moves

## 🎨 Color Selection

- **⚪ WHITE**: You play as White (move first)
- **⚫ BLACK**: You play as Black (respond to White)

**Board automatically flips** to show your perspective!

## 📊 Statistics Dashboard

Real-time stats updated every 2 seconds:

- **GPU STATUS**: Shows "METAL ACTIVE" on Apple Silicon
- **MODEL VERSION**: Current model version (increments with training)
- **MOVE COUNT**: Total moves in current game
- **GAME TIME**: Elapsed time since game start

## 🔥 Engine Logs

Live feed showing:
- Player moves
- Engine moves
- Thinking process
- Game events (checkmate, draw, etc.)
- Model reloads
- Simulation status

**Color-coded:**
- 🟢 Green = System messages
- 🔵 Cyan = Info messages
- 🟡 Yellow = Warnings
- 🔴 Red = Errors

## 📜 Move History

Scrollable list of all moves:
- Move number
- Player name
- Move notation

**Example:**
```
1. You: e4
2. BugzyEngine: e5
3. You: Nf3
4. BugzyEngine: Nc6
```

## 🎛️ Controls

### NEW GAME 🔄
- Resets the board
- Clears move history
- Starts fresh timer

### START/STOP SIMULATION ▶️⏹️
- Only visible in Simulation Mode
- Toggles auto-play on/off
- Green when stopped, red when running

### COPY PGN 📋
- Exports game in standard PGN format
- Copies to clipboard
- Includes headers (date, players, result)

## 🚀 Quick Start Guide

### Play Against BugzyEngine

```bash
cd ~/BugzyEngine
git pull origin master
./stop.sh
./run.sh
```

Open http://localhost:9443

1. Keep "Manual Play" selected
2. Choose your color
3. Start moving pieces!

### Run Simulation

1. Select "Simulation Mode"
2. Adjust Stockfish ELO slider
3. Click "START SIMULATION"
4. Watch the battle!

### Get Move Hints

1. Play in Manual Mode
2. After each move, check "Move Suggestions" panel
3. See top 3 moves with evaluations
4. Learn from the engine!

## 🔧 Technical Details

### Stockfish Integration

- **Engine**: Uses python-chess library
- **UCI Protocol**: Standard chess engine communication
- **ELO Limiting**: `UCI_LimitStrength` and `UCI_Elo` options
- **Time Control**: 0.1 seconds per move (fast gameplay)

### Move Suggestions

- **Analysis Depth**: 15 ply (half-moves)
- **Multi-PV**: Top 3 variations
- **Evaluation**: Centipawn scores from Stockfish
- **Format**: Standard Algebraic Notation (SAN)

### Hot-Reload System

- Model checks for updates every 2 seconds
- Automatically reloads when trainer saves new version
- Zero-downtime model switching
- Version tracking in GUI

## 🎯 Use Cases

### 1. Training
- Play against BugzyEngine at different stages of training
- Watch it improve as model version increments
- Use move suggestions to learn optimal play

### 2. Benchmarking
- Run simulations at different Stockfish ELO levels
- Track win/loss/draw rates
- Measure BugzyEngine's strength objectively

### 3. Analysis
- Use move suggestions to analyze positions
- Compare BugzyEngine's moves to Stockfish's recommendations
- Identify weaknesses in your AI's play

### 4. Entertainment
- Watch epic battles between BugzyEngine and Stockfish
- Challenge yourself against the AI
- Learn chess strategies from move suggestions

## 🐛 Troubleshooting

### "Stockfish not found"
```bash
brew install stockfish
```

### Simulation not starting
- Check that Stockfish is installed
- Verify logs for error messages
- Restart the web GUI

### Move suggestions not showing
- Stockfish must be installed
- Check engine logs for errors
- Ensure game is not over

### Board not updating in simulation
- Refresh browser
- Check simulation status (should show "running")
- Verify both engines are working

## 📈 Performance Tips

### For Faster Simulations
- Lower Stockfish ELO (faster computation)
- Reduce analysis depth (modify code)
- Use faster hardware

### For Better Analysis
- Increase Stockfish ELO
- Increase analysis depth
- Allow more time per move

## 🎉 What's Next?

Future enhancements:
- **Opening book integration**: Use specialized opening model for first 10 moves
- **Multiple simultaneous simulations**: Run 10+ games in parallel
- **Statistics dashboard**: Win rates, average moves, ELO estimation
- **Game database**: Save and replay games
- **Tournament mode**: Round-robin against multiple ELO levels

---

**BugzyEngine v5.0** - The Complete Chess AI Training Platform
