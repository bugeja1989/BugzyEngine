#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Start the processes
echo "Starting BugzyEngine..."

# Start the data collector in the background
echo "Starting Data Collector..."
python3 process1_chesscom_collector.py & 
COLLECTOR_PID=$!

# Start the trainer in the background
echo "Starting Neural Network Trainer..."
python3 process2_trainer.py & 
TRAINER_PID=$!

# Start the web GUI in the foreground
echo "Starting Web GUI on http://localhost:9443"
python3 web_gui.py

# Clean up background processes on exit
kill $COLLECTOR_PID
kill $TRAINER_PID
echo "BugzyEngine stopped."
