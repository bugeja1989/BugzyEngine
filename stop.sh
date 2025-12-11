#!/bin/bash

# BugzyEngine Shutdown Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bugzy.pid"

echo "========================================="
echo "    🛑 Stopping BugzyEngine"
echo "========================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ BugzyEngine is not running (no PID file found)."
    exit 1
fi

# Read PIDs
COLLECTOR_PID=$(sed -n '1p' "$PID_FILE")
TRAINER_PID=$(sed -n '2p' "$PID_FILE")
WEBGUI_PID=$(sed -n '3p' "$PID_FILE")

echo "📊 Stopping processes..."

# Stop Collector
if kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    echo "   └─ Stopping Collector (PID: $COLLECTOR_PID)..."
    kill "$COLLECTOR_PID"
else
    echo "   └─ Collector already stopped"
fi

# Stop Trainer
if kill -0 "$TRAINER_PID" 2>/dev/null; then
    echo "   └─ Stopping Trainer (PID: $TRAINER_PID)..."
    kill "$TRAINER_PID"
else
    echo "   └─ Trainer already stopped"
fi

# Stop Web GUI
if kill -0 "$WEBGUI_PID" 2>/dev/null; then
    echo "   └─ Stopping Web GUI (PID: $WEBGUI_PID)..."
    kill "$WEBGUI_PID"
else
    echo "   └─ Web GUI already stopped"
fi

# Wait a moment for graceful shutdown
sleep 2

# Force kill if still running
for PID in "$COLLECTOR_PID" "$TRAINER_PID" "$WEBGUI_PID"; do
    if kill -0 "$PID" 2>/dev/null; then
        echo "   └─ Force killing PID: $PID"
        kill -9 "$PID" 2>/dev/null
    fi
done

# Remove PID file
rm -f "$PID_FILE"

echo ""
echo "✅ BugzyEngine stopped successfully!"
echo ""
