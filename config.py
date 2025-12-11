#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py: Centralized configuration for BugzyEngine.
"""

# --- System --- 
WEB_PORT = 9443
GPU_DEVICE = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" for CPU-only

# --- Collector --- 
CHESS_COM_API_BASE_URL = "https://api.chess.com/pub"
HISTORICAL_LEGENDS = [
    "Mikhail-Tal", "Bobby-Fischer", "Garry-Kasparov", "Paul-Morphy",
    "Alexander-Alekhine", "Mikhail-Botvinnik", "Jose-Raul-Capablanca",
    "Leonid-Stein", "David-Bronstein", "Judit-Polgar", "Magnus-Carlsen", "Hikaru"
]
COLLECTOR_CONCURRENCY = 20  # Number of players to process in parallel
ELO_THRESHOLD = 2600  # Minimum ELO for titled players
TOP_N_PLAYERS = 100 # Number of top players to fetch from leaderboards/titled lists

# --- Filtering ---
MIN_COMPLEXITY_SCORE = 15 # (captures + checks)
MIN_SACRIFICES = 1
MIN_DRAW_COMPLEXITY = 25
MIN_ACCURACY_FOR_DRAW = 0.95 # Placeholder, as accuracy is not in PGN

# --- Trainer ---
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 6

# --- Search ---
SEARCH_DEPTH = 3
