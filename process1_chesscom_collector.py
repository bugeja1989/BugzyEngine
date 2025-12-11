#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process1_chesscom_collector.py: High-performance data collector for the BugzyEngine.
"""

import asyncio
import aiohttp
import os
import time
import chess.pgn
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from logging_config import setup_logging
from config import (
    HISTORICAL_LEGENDS, CHESS_COM_API_BASE_URL, COLLECTOR_CONCURRENCY, ELO_THRESHOLD, 
    TOP_N_PLAYERS, MIN_COMPLEXITY_SCORE, MIN_SACRIFICES, MIN_DRAW_COMPLEXITY, MIN_ACCURACY_FOR_DRAW
)

DATA_DIR = "/home/ubuntu/BugzyEngine/data/raw_pgn"
DB_PATH = "/home/ubuntu/BugzyEngine/data/collector.db"

# --- Database Setup ---
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS processed_games 
                     (game_id TEXT PRIMARY KEY, player TEXT, downloaded_at TIMESTAMP)""")
        conn.commit()

def is_game_processed(game_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT 1 FROM processed_games WHERE game_id = ?", (game_id,))
        return c.fetchone() is not None

def mark_game_processed(game_id, player):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO processed_games VALUES (?, ?, CURRENT_TIMESTAMP)", (game_id, player))
        conn.commit()

# --- Async Downloader ---
async def fetch(session, url):
    retries = 3
    for i in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429: # Rate limit
                    await asyncio.sleep(2 ** i) # Exponential backoff
                else:
                    logger.error(f"Error fetching {url}: {response.status}")
                    return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error fetching {url}: {e}")
            await asyncio.sleep(1)
    return None

async def fetch_pgn(session, url):
    retries = 3
    for i in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 429:
                    await asyncio.sleep(2 ** i)
                else:
                    return None
        except aiohttp.ClientError:
            await asyncio.sleep(1)
    return None

# --- Filtering Logic ---
def get_material_balance(board):
    balance = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9
            }.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value
    return balance

def is_game_style_match(game, player_username):
    """Advanced filter for game style: wins, sacrifices, complexity."""
    result = game.headers.get("Result", "*")
    white_player = game.headers.get("White", "").lower()
    black_player = game.headers.get("Black", "").lower()

    player_color = chess.WHITE if player_username.lower() == white_player else chess.BLACK

    # 1. Win/Draw Condition
    player_won = (result == "1-0" and player_color == chess.WHITE) or (result == "0-1" and player_color == chess.BLACK)
    is_draw = result == "1/2-1/2"

    # Accuracy check for draws is a placeholder for now
        if not player_won and not (is_draw and game.headers.get("Accuracy", 0) > MIN_ACCURACY_FOR_DRAW):
        return False

    # 2. Complexity and Sacrifice Heuristics
    board = game.board()
    initial_material = get_material_balance(board)
    sacrifices_made = 0
    checks_made = 0
    captures_made = 0

    for move in game.mainline_moves():
        is_capture = board.is_capture(move)
        if is_capture:
            captures_made += 1
        
        if board.gives_check(move):
            checks_made += 1

        # Sacrifice heuristic
        if is_capture and board.color_at(move.from_square) == player_color:
            pre_move_material = get_material_balance(board)
            board.push(move)
            post_move_material = get_material_balance(board)
            # If player's material decreased after a capture, it could be a sacrifice
            if (player_color == chess.WHITE and post_move_material < pre_move_material) or \
               (player_color == chess.BLACK and post_move_material > pre_move_material):
                sacrifices_made += 1
        else:
            board.push(move)

    complexity_score = captures_made + checks_made

    # 3. Final Decision
        if player_won and (sacrifices_made >= MIN_SACRIFICES or complexity_score > MIN_COMPLEXITY_SCORE):
        return True
    
    # Keep high-complexity draws (placeholder for accuracy)
        if is_draw and complexity_score > MIN_DRAW_COMPLEXITY:
        return True

    return False

# --- Main Collector Logic ---
async def process_player(session, player_username):
    logger.info(f"Processing player: {player_username}")
    archives_url = f"{CHESS_COM_API_BASE_URL}/player/{player_username}/games/archives"
    archives_json = await fetch(session, archives_url)
    if not archives_json or "archives" not in archives_json:
        return

    for archive_url in archives_json["archives"]:
        pgn_text = await fetch_pgn(session, archive_url + "/pgn")
        if not pgn_text:
            continue

        with open(f"{DATA_DIR}/{player_username}_temp.pgn", "w") as f:
            f.write(pgn_text)

        with open(f"{DATA_DIR}/{player_username}_temp.pgn") as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                    if game is None: break
                    game_id = game.headers.get("Link", "").split("/")[-1]
                    if not game_id or is_game_processed(game_id):
                        continue

                    if is_game_style_match(game, player_username):
                        with open(f"{DATA_DIR}/{player_username}_{game_id}.pgn", "w") as out_pgn:
                            exporter = chess.pgn.FileExporter(out_pgn)
                            game.accept(exporter)
                        mark_game_processed(game_id, player_username)
                except Exception as e:
                    logger.error(f"Malformed game in {player_username}'s archive: {e}")
                    continue
        os.remove(f"{DATA_DIR}/{player_username}_temp.pgn")

async def main_async():
    os.makedirs(DATA_DIR, exist_ok=True)
    init_db()
    semaphore = asyncio.Semaphore(COLLECTOR_CONCURRENCY)

    async def process_player_sema(session, player_username):
        async with semaphore:
            return await process_player(session, player_username)

    async with aiohttp.ClientSession(headers={"User-Agent": "BugzyEngine/1.0"}) as session:
        # --- Dynamic Player Discovery ---
        target_players = set(HISTORICAL_LEGENDS)
        logger.info("Starting dynamic player discovery...")

        # 1. Fetch Leaderboards
        for category in ["live_blitz", "live_bullet", "live_rapid"]:
            leaderboard_data = await fetch(session, f"{CHESS_COM_API_BASE_URL}/leaderboards")
            if leaderboard_data and category in leaderboard_data:
                for player in leaderboard_data[category][:TOP_N_PLAYERS]:
                    target_players.add(player["username"])

        # 2. Fetch Titled Players (GM, IM)
        for title in ["GM", "IM"]:
            titled_players = await fetch(session, f"{CHESS_COM_API_BASE_URL}/titled/{title}")
            if titled_players and "players" in titled_players:
                # This part is slow as it checks ELO for each player individually
                # A full implementation would need a more efficient way to get ELOs
                for player_username in titled_players["players"][:TOP_N_PLAYERS * 2]: # Fetch more to filter down
                    stats = await fetch(session, f"{CHESS_COM_API_BASE_URL}/player/{player_username}/stats")
                    if stats and stats.get("chess_rapid", {}).get("last", {}).get("rating", 0) > ELO_THRESHOLD:
                        target_players.add(player_username)
                    await asyncio.sleep(0.1) # Be nice to the API

        logger.info(f"Discovered {len(target_players)} unique players to process.")

        # --- Concurrent Game Collection ---
        tasks = [process_player_sema(session, player) for player in target_players]
        await asyncio.gather(*tasks)

# --- Folder Watcher ---
class PGNHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pgn"):
            logger.info(f"New local PGN file detected: {event.src_path}")

def main():
    # Start folder watcher in a separate thread (not implemented yet)
    observer = Observer()
    observer.schedule(PGNHandler(), path=DATA_DIR, recursive=False)
    observer.start()
    logger.info(f"Watching for new .pgn files in {DATA_DIR}")

    # Run async collector
    asyncio.run(main_async())

    try:
        while True:
            time.sleep(3600) # Re-run collection every hour
            asyncio.run(main_async())
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

logger = setup_logging("Collector", "collector.log")

if __name__ == "__main__":
    main()
