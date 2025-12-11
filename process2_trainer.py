'''
process2_trainer.py: Continuous trainer for the BugzyEngine with parallel batch processing and caching.
'''

import os
import time
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import Pool, cpu_count
from neural_network.src.engine_utils import board_to_tensor
from config import GPU_DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS
from logging_config import setup_logging

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "raw_pgn")
CACHE_PATH = os.path.join(SCRIPT_DIR, "data", "cache")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bugzy_model.pth")

# Hybrid Training Settings
BATCH_PROCESS_SIZE = 1000  # Process 1000 files at a time
NUM_WORKERS = cpu_count()  # Use all available CPU cores

device = torch.device(GPU_DEVICE)

# --- Model Definition ---
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

# --- Tier 1: Smart Caching & Parallel Processing ---
def process_pgn_file(file_path):
    '''Processes a single PGN file and returns positions and outcomes.'''
    cache_file = os.path.join(CACHE_PATH, os.path.basename(file_path) + ".npz")
    
    # Use cache if it exists
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            return list(data['positions']), list(data['outcomes'])
        except Exception as e:
            # logger.warning(f"Corrupt cache file {cache_file}, re-processing. Error: {e}")
            pass

    # Process PGN if not cached
    positions = []
    outcomes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as pgn:
            game = chess.pgn.read_game(pgn)
            if not game:
                return [], []

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
        return positions, outcomes

    except Exception as e:
        # logger.warning(f"Skipping malformed PGN {file_path}: {e}")
        return [], []

# --- Tier 2: Batch Training ---
def train_on_batch(model, file_batch):
    '''Trains the model on a batch of PGN files in parallel.'''
    logger.info(f"🧠 Processing batch of {len(file_batch)} files with {NUM_WORKERS} workers...")
    
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_pgn_file, file_batch)

    all_positions = []
    all_outcomes = []
    for positions, outcomes in results:
        all_positions.extend(positions)
        all_outcomes.extend(outcomes)

    if not all_positions:
        logger.info("No new training data in this batch.")
        return

    logger.info(f"🚀 Training on {len(all_positions):,} new positions from this batch.")
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
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        logger.info(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader):.6f}")

    # Atomic save with version tracking
    temp_path = MODEL_PATH + ".tmp"
    torch.save(model.state_dict(), temp_path)
    os.rename(temp_path, MODEL_PATH)
    
    version_file = os.path.join(os.path.dirname(MODEL_PATH), "version.txt")
    try:
        with open(version_file, "r") as f: version = int(f.read().strip()) + 1
    except: version = 1
    with open(version_file, "w") as f: f.write(str(version))
    
    logger.info(f"✅ Model v{version} saved! Now live in the GUI.")

# --- Main Loop ---
def main():
    '''Main function for the trainer.'''
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model = ChessNet().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Loaded existing model.")

    processed_files = set()

    while True:
        logger.info("Scanning for new PGN files...")
        try:
            all_pgn_files = {os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".pgn")}
            new_files = sorted(list(all_pgn_files - processed_files))

            if not new_files:
                logger.info("No new games found. Waiting...")
                time.sleep(30)
                continue

            logger.info(f"🔥 Found {len(new_files):,} new PGN files. Starting training cycles...")

            for i in range(0, len(new_files), BATCH_PROCESS_SIZE):
                batch = new_files[i:i + BATCH_PROCESS_SIZE]
                train_on_batch(model, batch)
                processed_files.update(batch)
                logger.info(f"Completed batch {i // BATCH_PROCESS_SIZE + 1}/{(len(new_files) + BATCH_PROCESS_SIZE - 1) // BATCH_PROCESS_SIZE}. Model is getting smarter!")

        except Exception as e:
            logger.error(f"An error occurred in the training loop: {e}", exc_info=True)
            time.sleep(60)

if __name__ == "__main__":
    logger = setup_logging('trainer')
    logger.info("BugzyEngine Trainer v4.0 with Hybrid Training started.")
    main()
