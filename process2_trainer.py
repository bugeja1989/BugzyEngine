#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process2_trainer.py: Continuous trainer for the BugzyEngine.
"""

import os
import time
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network.src.engine_utils import board_to_tensor
from config import GPU_DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS
from logging_config import setup_logging

DATA_PATH = "/home/ubuntu/BugzyEngine/data/raw_pgn"
PROCESSED_PATH = "/home/ubuntu/BugzyEngine/data/processed"
MODEL_PATH = "/home/ubuntu/BugzyEngine/models/bugzy_model.pth"

device = torch.device(GPU_DEVICE)

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x

def train_model(model, data_path):
    """Trains the neural network model."""
    positions = []
    outcomes = []
    logger.info(f"Scanning for PGN files in {data_path}")
    for pgn_file in os.listdir(data_path):
        if pgn_file.endswith(".pgn"):
            logger.info(f"Processing {pgn_file}...")
            with open(os.path.join(data_path, pgn_file)) as f:
                while True:
                    try:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        result = game.headers.get("Result", "*")
                        if result == "1-0":
                            outcome = 1.0
                        elif result == "0-1":
                            outcome = -1.0
                        else:
                            continue # Skip draws and unfinished games

                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                            positions.append(board_to_tensor(board.copy()))
                            outcomes.append(outcome)
                    except Exception as e:
                        logger.warning(f"Skipping a malformed game in {pgn_file}: {e}")
                        continue

    if not positions:
        logger.info("No new training data found.")
        return

    logger.info(f"Training on {len(positions)} new positions.")
    X = torch.cat(positions)
    y = torch.FloatTensor(outcomes).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Atomic save
    temp_path = MODEL_PATH + ".tmp"
    torch.save(model.state_dict(), temp_path)
    os.rename(temp_path, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

def main():
    """Main function for the trainer."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = ChessNet().to(device)
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    logger.info("BugzyEngine Trainer started.")
    logger.info(f"Using device: {device}")

    while True:
        logger.info("Watching for new PGN files...")
        found_new_files = False
        pgn_files_in_dir = [f for f in os.listdir(DATA_PATH) if f.endswith(".pgn")]
        if pgn_files_in_dir:
            logger.info(f"Found {len(pgn_files_in_dir)} PGN files. Starting training cycle.")
            train_model(model, DATA_PATH)
            # Move all processed files
            for pgn_file in pgn_files_in_dir:
                src_path = os.path.join(DATA_PATH, pgn_file)
                dest_path = os.path.join(PROCESSED_PATH, pgn_file)
                shutil.move(src_path, dest_path)
            logger.info(f"Moved {len(pgn_files_in_dir)} processed files to {PROCESSED_PATH}")
        else:
            time.sleep(60) # Wait for a minute before checking again

logger = setup_logging("Trainer", "trainer.log")

if __name__ == "__main__":
    import shutil
    main()
