#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
process2_trainer.py: BugzyEngine v4.1 - Hybrid Training with Verbose Logging
Features:
- Parallel batch processing with multiprocessing
- Smart caching system (PGN → numpy arrays)
- Incremental training (train after each batch)
- Verbose logging with statistics
- Skip empty batches intelligently
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
from config import GPU_DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS
from logging_config import setup_logging

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "raw_pgn")
CACHE_PATH = os.path.join(SCRIPT_DIR, "data", "cache")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bugzy_model.pth")

device = torch.device(GPU_DEVICE)

# Hybrid Training Parameters
NUM_WORKERS = 14  # Parallel workers for file processing
BATCH_PROCESS_SIZE = 1000  # Files per training cycle
MAX_EMPTY_BATCHES = 5  # Stop after X consecutive empty batches

# --- Model Architecture ---
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

# --- Tier 1: Smart Caching ---
def process_single_pgn(file_path):
    '''Process a single PGN file with caching.'''
    cache_file = os.path.join(CACHE_PATH, os.path.basename(file_path) + ".npz")
    
    # Check cache first
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            positions = list(data['positions'])
            outcomes = list(data['outcomes'])
            return file_path, positions, outcomes, True  # cached=True
        except Exception as e:
            pass  # Re-process if cache corrupt

    # Process PGN if not cached
    positions = []
    outcomes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as pgn:
            game = chess.pgn.read_game(pgn)
            if not game:
                # Save empty cache to avoid re-processing
                np.savez_compressed(cache_file, positions=np.array([]), outcomes=np.array([]))
                return file_path, [], [], False
            
            result = game.headers.get("Result", "*")
            if result == "1-0": outcome = 1.0
            elif result == "0-1": outcome = -1.0
            else: outcome = 0.0

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board_to_tensor(board.copy()).numpy())
                outcomes.append(outcome)
        
        # Save to cache
        np.savez_compressed(cache_file, positions=np.array(positions), outcomes=np.array(outcomes))
        return file_path, positions, outcomes, False  # cached=False

    except Exception as e:
        # Save empty cache for malformed files
        np.savez_compressed(cache_file, positions=np.array([]), outcomes=np.array([]))
        return file_path, [], [], False

# --- Tier 2: Batch Training ---
def train_on_batch(model, file_batch, batch_num, total_batches):
    '''Trains the model on a batch of PGN files in parallel.'''
    logger.info(f"📦 Batch {batch_num}/{total_batches}: Processing {len(file_batch)} files with {NUM_WORKERS} workers...")
    
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_single_pgn, file_batch)
    
    # Collect statistics
    all_positions = []
    all_outcomes = []
    cached_count = 0
    processed_count = 0
    empty_count = 0
    
    for file_path, positions, outcomes, cached in results:
        if cached:
            cached_count += 1
        else:
            processed_count += 1
        
        if not positions:
            empty_count += 1
        else:
            all_positions.extend(positions)
            all_outcomes.extend(outcomes)
    
    # Log statistics
    logger.info(f"  📊 Stats: {cached_count} cached | {processed_count} processed | {empty_count} empty")
    logger.info(f"  📈 Found {len(all_positions):,} positions from {len(file_batch) - empty_count} valid games")

    if not all_positions:
        logger.info(f"  ⏭️  Skipping training - no new positions in this batch")
        return False  # No training occurred
    
    # Train on collected positions
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
        batch_count = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        logger.info(f"    Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}")

    # Atomic save with version tracking
    temp_path = MODEL_PATH + ".tmp"
    torch.save(model.state_dict(), temp_path)
    os.rename(temp_path, MODEL_PATH)
    
    version_file = os.path.join(os.path.dirname(MODEL_PATH), "version.txt")
    try:
        with open(version_file, "r") as f: version = int(f.read().strip()) + 1
    except: version = 1
    with open(version_file, "w") as f: f.write(str(version))
    
    logger.info(f"  ✅ Model v{version} saved and live!")
    return True  # Training occurred

# --- Main Loop ---
def main():
    '''Main function for the trainer.'''
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = ChessNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("✅ Loaded existing model")
    else:
        logger.info("🆕 No existing model - will create new one")

    processed_files = set()
    consecutive_empty_batches = 0

    while True:
        logger.info("🔍 Scanning for new PGN files...")
        try:
            all_pgn_files = {os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pgn")}
            new_files = sorted(list(all_pgn_files - processed_files))

            if not new_files:
                logger.info("⏸️  No new games found. Waiting 30 seconds...")
                time.sleep(30)
                consecutive_empty_batches = 0  # Reset counter
                continue

            logger.info(f"🔥 Found {len(new_files):,} new PGN files!")
            total_batches = (len(new_files) + BATCH_PROCESS_SIZE - 1) // BATCH_PROCESS_SIZE
            logger.info(f"📋 Will process in {total_batches} batches of {BATCH_PROCESS_SIZE} files each")

            for i in range(0, len(new_files), BATCH_PROCESS_SIZE):
                batch = new_files[i:i + BATCH_PROCESS_SIZE]
                batch_num = i // BATCH_PROCESS_SIZE + 1
                
                trained = train_on_batch(model, batch, batch_num, total_batches)
                processed_files.update(batch)
                
                if trained:
                    consecutive_empty_batches = 0
                    logger.info(f"✨ Batch {batch_num}/{total_batches} complete! Model improved.")
                else:
                    consecutive_empty_batches += 1
                    logger.info(f"⏭️  Batch {batch_num}/{total_batches} skipped (no data). Empty streak: {consecutive_empty_batches}")
                
                # Stop if too many consecutive empty batches
                if consecutive_empty_batches >= MAX_EMPTY_BATCHES:
                    logger.info(f"🛑 Stopping after {MAX_EMPTY_BATCHES} consecutive empty batches")
                    logger.info(f"💾 All {len(processed_files):,} files have been processed and cached")
                    break
            
            # Reset counter after completing all batches
            if consecutive_empty_batches < MAX_EMPTY_BATCHES:
                logger.info(f"🎉 Completed processing all {len(new_files):,} new files!")
                consecutive_empty_batches = 0

        except Exception as e:
            logger.error(f"❌ Error in training loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    logger = setup_logging('Trainer', 'trainer.log')
    logger.info("🚀 BugzyEngine Trainer v4.1 with Verbose Logging started")
    logger.info(f"⚙️  Config: {NUM_WORKERS} workers | Batch size: {BATCH_PROCESS_SIZE} | Epochs: {EPOCHS}")
    logger.info(f"🎯 Device: {device}")
    main()
