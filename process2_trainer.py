#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process2_trainer.py: BugzyEngine v5.2 - Two-Tier Filtering
"""

import os
import time
import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import Pool
from neural_network.src.engine_utils import board_to_tensor
from config import (
    GPU_DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, 
    TIER2_ENABLED, TIER2_MIN_COMPLEXITY, TIER2_MIN_SACRIFICES, TIER2_MIN_DRAW_COMPLEXITY
)
from logging_config import setup_logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "raw_pgn")
CACHE_PATH = os.path.join(SCRIPT_DIR, "data", "cache")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bugzy_model.pth")

device = torch.device(GPU_DEVICE)

NUM_WORKERS = 14
BATCH_PROCESS_SIZE = 1000
MAX_EMPTY_BATCHES = 5

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

def tier2_style_filter(game):
    """Advanced filter for attacking style."""
    if not TIER2_ENABLED:
        return True

    board = game.board()
    sacrifices = 0
    checks = 0
    captures = 0

    for move in game.mainline_moves():
        if board.is_capture(move): captures += 1
        if board.gives_check(move): checks += 1
        board.push(move)

    complexity = captures + checks
    result = game.headers.get("Result", "*")
    is_draw = result == "1/2-1/2"

    if is_draw and complexity >= TIER2_MIN_DRAW_COMPLEXITY:
        return True
    if not is_draw and complexity >= TIER2_MIN_COMPLEXITY:
        return True

    return False

def process_single_pgn(file_path):
    cache_file = os.path.join(CACHE_PATH, os.path.basename(file_path) + ".npz")
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            return file_path, list(data["positions"]), list(data["outcomes"]), True
        except Exception:
            pass

    positions, outcomes = [], []
    try:
        with open(file_path, "r", encoding="utf-8") as pgn:
            game = chess.pgn.read_game(pgn)
            if game and tier2_style_filter(game):
                result = game.headers.get("Result", "*")
                outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    positions.append(board_to_tensor(board.copy()).cpu().numpy())
                    outcomes.append(outcome)
    except Exception:
        pass
    
    # Only cache if we have valid positions
    if positions:
        np.savez_compressed(cache_file, positions=np.array(positions), outcomes=np.array(outcomes))
    return file_path, positions, outcomes, False

def train_on_batch(model, file_batch, batch_num, total_batches):
    logger.info(f"📦 Batch {batch_num}/{total_batches}: Processing {len(file_batch)} files...")
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_single_pgn, file_batch)
    
    all_positions, all_outcomes, cached_count, processed_count, empty_count = [], [], 0, 0, 0
    for _, positions, outcomes, cached in results:
        if cached: cached_count += 1
        else: processed_count += 1
        if not positions: empty_count += 1
        else:
            all_positions.extend(positions)
            all_outcomes.extend(outcomes)

    logger.info(f"  📊 Stats: {cached_count} cached | {processed_count} processed | {empty_count} empty")
    logger.info(f"  📈 Found {len(all_positions):,} positions from {len(file_batch) - empty_count} valid games")

    if not all_positions:
        logger.info("  ⏭️  Skipping training - no new positions")
        return False

    logger.info(f"  🚀 Training on {len(all_positions):,} positions...")
    X = torch.from_numpy(np.array(all_positions)).to(device)
    y = torch.FloatTensor(all_outcomes).unsqueeze(1).to(device)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"    Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.6f}")

    temp_path = MODEL_PATH + ".tmp"
    torch.save(model.state_dict(), temp_path)
    os.rename(temp_path, MODEL_PATH)
    
    version_file = os.path.join(os.path.dirname(MODEL_PATH), "version.txt")
    try: version = int(open(version_file).read().strip()) + 1
    except: version = 1
    with open(version_file, "w") as f: f.write(str(version))
    
    logger.info(f"  ✅ Model v{version} saved!")
    return True

def main():
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = ChessNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("✅ Loaded existing model")
    else:
        logger.info("🆕 No existing model")

    processed_files = set()
    consecutive_empty_batches = 0

    while True:
        logger.info("🔍 Scanning for new PGN files...")
        try:
            all_pgn_files = {os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pgn")}
            new_files = sorted(list(all_pgn_files - processed_files))

            if not new_files:
                logger.info("⏸️  No new games found. Waiting 30s...")
                time.sleep(30)
                continue

            logger.info(f"🔥 Found {len(new_files):,} new PGN files!")
            total_batches = (len(new_files) + BATCH_PROCESS_SIZE - 1) // BATCH_PROCESS_SIZE
            logger.info(f"📋 Will process in {total_batches} batches")

            for i in range(0, len(new_files), BATCH_PROCESS_SIZE):
                batch = new_files[i:i + BATCH_PROCESS_SIZE]
                batch_num = i // BATCH_PROCESS_SIZE + 1
                
                trained = train_on_batch(model, batch, batch_num, total_batches)
                processed_files.update(batch)
                
                if trained: consecutive_empty_batches = 0
                else: consecutive_empty_batches += 1
                
                if consecutive_empty_batches >= MAX_EMPTY_BATCHES:
                    logger.info(f"🛑 Stopping after {MAX_EMPTY_BATCHES} empty batches")
                    break
            
            logger.info(f"🎉 Completed processing all {len(new_files):,} new files!")

        except Exception as e:
            logger.error(f"❌ Error in training loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    logger = setup_logging("Trainer", "trainer.log")
    logger.info("🚀 BugzyEngine Trainer v5.2 with Two-Tier Filtering started")
    logger.info(f"🎯 Device: {device}")
    main()
