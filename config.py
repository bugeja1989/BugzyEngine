#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py: Configuration for the BugzyEngine.
"""

# Web GUI
WEB_PORT = 9443

# Neural Network
GPU_DEVICE = "mps"

# Hyperparameters
HISTORICAL_LEGENDS = [
    "Mikhail-Tal", "Bobby-Fischer", "Garry-Kasparov", "Paul-Morphy", 
    "Alexander-Alekhine", "Mikhail-Botvinnik", "Jose-Raul-Capablanca",
    "Leonid-Stein", "David-Bronstein", "Judit-Polgar", "Magnus-Carlsen", "Hikaru-Nakamura"
]

CHESS_COM_API_BASE_URL = "https://api.chess.com/pub"
