# BugzyEngine Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-12-11

### 🚀 Major Enhancements

#### Async/Parallel Data Collection
- **Replaced synchronous requests with `aiohttp`** for concurrent game downloads
- **Semaphore-based concurrency control** (configurable via `COLLECTOR_CONCURRENCY`)
- **Exponential backoff** for rate limit handling (429 errors)
- **Database tracking** with SQLite to prevent duplicate game downloads
- **Resume capability**: Automatically skips already-processed games

#### Advanced "Greatness" Filter
- **Material balance tracking** to detect sacrifices
- **Complexity scoring** based on captures and checks
- **Sacrifice heuristic**: Identifies material sacrifices for initiative
- **Configurable thresholds**:
  - `MIN_COMPLEXITY_SCORE`: Minimum captures + checks (default: 15)
  - `MIN_SACRIFICES`: Minimum sacrifices required (default: 1)
  - `MIN_DRAW_COMPLEXITY`: Complexity threshold for draws (default: 25)
  - `MIN_ACCURACY_FOR_DRAW`: Accuracy threshold for draws (default: 0.95)

#### Dynamic Player Discovery
- **Leaderboard integration**: Fetches top players from Blitz, Bullet, and Rapid leaderboards
- **Titled player filtering**: Automatically discovers GMs and IMs with ELO > 2600
- **Configurable target**: `TOP_N_PLAYERS` (default: 100)
- **Historical legends**: Always includes Tal, Fischer, Kasparov, Morphy, Carlsen, Nakamura, etc.

#### Centralized Logging System
- **New `logging_config.py` module** for unified logging across all processes
- **Separate log files** for collector and trainer (`collector.log`, `trainer.log`)
- **Console + file output** with timestamps and log levels
- **Structured logging** for better debugging and monitoring

#### Enhanced Configuration
- **Expanded `config.py`** with comprehensive settings:
  - Collector settings (concurrency, ELO threshold, top N players)
  - Filtering thresholds (complexity, sacrifices, accuracy)
  - Trainer hyperparameters (learning rate, batch size, epochs)
  - Search depth configuration
- **Centralized hyperparameters**: All tunable values in one place

### 🔧 Improvements

#### Data Collector (`process1_chesscom_collector.py`)
- **Async game processing** with `asyncio.gather()` for parallel downloads
- **Smart rate limiting** with 0.1s delays between player stat checks
- **Better error handling** with retry logic and exponential backoff
- **Folder watcher** for local PGN file integration (TWIC, Lichess)
- **Database persistence** to track processed games

#### Neural Network Trainer (`process2_trainer.py`)
- **Config integration**: Uses `LEARNING_RATE`, `BATCH_SIZE`, and `EPOCHS` from config
- **Enhanced logging**: All print statements replaced with structured logging
- **Better error handling**: Warns on malformed games instead of crashing

#### Web GUI (`web_gui.py`)
- No changes in this version (already using shared `engine_utils.py`)

### 📦 Dependencies
- **Added `aiohttp`** for asynchronous HTTP requests
- **Added `watchdog`** for folder monitoring (already present)

### 🐛 Bug Fixes
- Fixed indentation issues in `is_game_style_match()` function
- Corrected string formatting in error messages (removed stray quotes)
- Improved material balance calculation for sacrifice detection

### 📝 Documentation
- Updated README with new features and configuration options
- Added CHANGELOG for version tracking

### 🔮 Future Enhancements
- Opening book integration (first 10 moves)
- Lichess API integration for additional data sources
- Real-time training dashboard
- UCI protocol support
- Multi-GPU training

---

## [1.0.0] - 2025-12-11

### Initial Release
- Basic data collector with Chess.com API integration
- PyTorch-based neural network trainer with MPS support
- Web GUI on Port 9443
- Shared chess logic in `engine_utils.py`
- Alpha-beta search with move ordering
- Atomic model saving
- Installation and launcher scripts
- Comprehensive documentation

---

**Repository**: https://github.com/bugeja1989/BugzyEngine  
**License**: MIT  
**Author**: Manus AI
