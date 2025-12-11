#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py: Centralized configuration for BugzyEngine v5.2
"""

# --- System ---
WEB_PORT = 9443
GPU_DEVICE = "mps"

# --- Collector ---
CHESS_COM_API_BASE_URL = "https://api.chess.com/pub"
HISTORICAL_LEGENDS = [
    "Mikhail-Tal", "Bobby-Fischer", "Garry-Kasparov", "Paul-Morphy",
    "Alexander-Alekhine", "Mikhail-Botvinnik", "Jose-Raul-Capablanca",
    "Leonid-Stein", "David-Bronstein", "Judit-Polgar", "Magnus-Carlsen", "Hikaru"
]
COLLECTOR_CONCURRENCY = 20
ELO_THRESHOLD = 2600
TOP_N_PLAYERS = 100

# --- Two-Tier Filtering (v5.2) ---

# Tier 1: Basic Quality Filter (applied in collector)
# Goal: Get a large volume of decent-quality games for initial training
# Pass Rate: ~70% of games
TIER1_MIN_MOVES = 10
TIER1_VALID_TERMINATIONS = ["won by resignation", "won on time", "won by checkmate", "game drawn"]

# Tier 2: Attacking Style Filter (applied in trainer, configurable)
# Goal: Select for aggressive, tactical, "Bugzy" style games
# Pass Rate: ~40% of Tier 1 games
TIER2_ENABLED = True
TIER2_MIN_COMPLEXITY = 8  # (captures + checks)
TIER2_MIN_SACRIFICES = 0  # Allow games without sacrifices
TIER2_MIN_DRAW_COMPLEXITY = 15

# --- Trainer ---
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 6

# --- Search ---
SEARCH_DEPTH = 3
