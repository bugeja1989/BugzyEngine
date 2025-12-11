#!/bin/bash

# BugzyEngine Launcher - Runs all processes in background

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bugzy.pid"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

# Create opening_books directory
mkdir -p "$SCRIPT_DIR/opening_books"

# Download opening book if not present
OPENING_BOOK="$SCRIPT_DIR/opening_books/gm2001.bin"
if [ ! -f "$OPENING_BOOK" ]; then
    echo "📚 Downloading opening book (12,000+ GM positions)..."
    curl -L -o "$OPENING_BOOK" "https://github.com/gmcheems-org/free-opening-books/raw/refs/heads/main/books/bin/gm2001.bin"
    if [ -f "$OPENING_BOOK" ]; then
        echo "   ✅ Opening book downloaded successfully!"
    else
        echo "   ⚠️  Failed to download opening book (will use fallback)"
    fi
fi

# Check if already running
if [ -f "$PID_FILE" ]; then
    echo "BugzyEngine appears to be already running (PID file exists)."
    echo "If this is incorrect, run: ./stop.sh"
    exit 1
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

echo "========================================="
echo "    🚀 Starting BugzyEngine v2.0"
echo "========================================="
echo ""

# Start Data Collector
echo "📥 Starting Data Collector..."
python3 "$SCRIPT_DIR/process1_chesscom_collector.py" > "$LOG_DIR/collector_stdout.log" 2>&1 &
COLLECTOR_PID=$!
echo "   └─ PID: $COLLECTOR_PID"

# Start Neural Network Trainer
echo "🧠 Starting Neural Network Trainer..."
python3 "$SCRIPT_DIR/process2_trainer.py" > "$LOG_DIR/trainer_stdout.log" 2>&1 &
TRAINER_PID=$!
echo "   └─ PID: $TRAINER_PID"

# Start Web GUI
echo "🌐 Starting Web GUI..."
python3 "$SCRIPT_DIR/web_gui.py" > "$LOG_DIR/webgui_stdout.log" 2>&1 &
WEBGUI_PID=$!
echo "   └─ PID: $WEBGUI_PID"

# Save PIDs to file
echo "$COLLECTOR_PID" > "$PID_FILE"
echo "$TRAINER_PID" >> "$PID_FILE"
echo "$WEBGUI_PID" >> "$PID_FILE"

echo ""
echo "========================================="
echo "✅ BugzyEngine is now running!"
echo "========================================="
echo ""
echo "📊 Process IDs:"
echo "   Collector: $COLLECTOR_PID"
echo "   Trainer:   $TRAINER_PID"
echo "   Web GUI:   $WEBGUI_PID"
echo ""
echo "🌐 Web Interface: http://localhost:9443"
echo ""
echo "📁 Logs:"
echo "   Collector: $LOG_DIR/collector.log"
echo "   Trainer:   $LOG_DIR/trainer.log"
echo "   Web GUI:   $LOG_DIR/webgui_stdout.log"
echo ""
echo "🛑 To stop: ./stop.sh"
echo "📊 To monitor: tail -f logs/*.log"
echo ""
