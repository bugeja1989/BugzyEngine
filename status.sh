#!/bin/bash

# BugzyEngine Status Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.bugzy.pid"

echo "========================================="
echo "    📊 BugzyEngine Status"
echo "========================================="
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "❌ BugzyEngine is not running (no PID file found)."
    echo ""
    exit 1
fi

# Read PIDs
COLLECTOR_PID=$(sed -n '1p' "$PID_FILE")
TRAINER_PID=$(sed -n '2p' "$PID_FILE")
WEBGUI_PID=$(sed -n '3p' "$PID_FILE")

echo "📊 Process Status:"
echo ""

# Check Collector
if kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    echo "   ✅ Collector:  Running (PID: $COLLECTOR_PID)"
else
    echo "   ❌ Collector:  Stopped (PID: $COLLECTOR_PID)"
fi

# Check Trainer
if kill -0 "$TRAINER_PID" 2>/dev/null; then
    echo "   ✅ Trainer:    Running (PID: $TRAINER_PID)"
else
    echo "   ❌ Trainer:    Stopped (PID: $TRAINER_PID)"
fi

# Check Web GUI
if kill -0 "$WEBGUI_PID" 2>/dev/null; then
    echo "   ✅ Web GUI:    Running (PID: $WEBGUI_PID)"
else
    echo "   ❌ Web GUI:    Stopped (PID: $WEBGUI_PID)"
fi

echo ""
echo "🌐 Web Interface: http://localhost:9443"
echo ""
echo "📁 Recent Logs:"
echo "   Collector: tail -20 logs/collector.log"
echo "   Trainer:   tail -20 logs/trainer.log"
echo ""
