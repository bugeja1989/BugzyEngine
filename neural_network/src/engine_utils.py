#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
engine_utils.py: Core chess logic for the BugzyEngine.
"""

import chess
import numpy as np
import torch
from typing import Optional, Tuple
from config import GPU_DEVICE

device = torch.device(GPU_DEVICE)

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board object to a tensor representation.
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            square = chess.square(i, j)
            piece = board.piece_at(square)
            if piece:
                piece_type = piece.piece_type
                color = piece.color
                # One-hot encode piece type and color
                layer = (piece_type - 1) * 2 + (0 if color == chess.WHITE else 1)
                tensor[layer, i, j] = 1
    return torch.from_numpy(tensor).to(device)

def evaluate_position(board: chess.Board, model) -> float:
    """
    Evaluates the current board position using the neural network.
    """
    if model is None:
        return 0.0
    tensor = board_to_tensor(board)
    with torch.no_grad():
        value = model(tensor)
    return value.item()

def alpha_beta_search(board: chess.Board, depth: int, model, alpha: float = -np.inf, beta: float = np.inf) -> Tuple[float, Optional[chess.Move]]:
    """
    Performs an alpha-beta search to find the best move.
    """
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model), None

    best_move = None
    # Move ordering: try captures first
    moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)

    if board.turn == chess.WHITE:
        max_eval = -np.inf
        for move in moves:
            board.push(move)
            # Repetition Detection
            if board.is_repetition(2):
                board.pop()
                continue
            eval, _ = alpha_beta_search(board, depth - 1, model, alpha, beta)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = np.inf
        for move in moves:
            board.push(move)
            # Repetition Detection
            if board.is_repetition(2):
                board.pop()
                continue
            eval, _ = alpha_beta_search(board, depth - 1, model, alpha, beta)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move
