#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process1_chesscom_collector.py: BugzyEngine v5.2 - Two-Tier Filtering
"""

import asyncio
import aiohttp
import os
import time
import io
import chess.pgn
import sqlite3
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from logging_config import setup_logging
from config import (
    HISTORICAL_LEGENDS, CHESS_COM_API_BASE_URL, COLLECTOR_CONCURRENCY, ELO_THRESHOLD, 
    TOP_N_PLAYERS, TIER1_MIN_MOVES, TIER1_VALID_TERMINATIONS
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "raw_pgn")
DB_PATH = os.path.join(SCRIPT_DIR, "data", "collector.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS processed_games (game_id TEXT PRIMARY KEY, player TEXT, downloaded_at TIMESTAMP)")
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

async def fetch(session, url):
    retries = 3
    for i in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    await asyncio.sleep(2 ** i)
                else:
                    return None
        except aiohttp.ClientError:
            await asyncio.sleep(1)
    return None

def tier1_quality_filter(game):
    """Basic quality filter for games."""
    if len(list(game.mainline_moves())) < TIER1_MIN_MOVES:
        return False
    termination = game.headers.get("Termination", "").lower()
    if not any(term in termination for term in TIER1_VALID_TERMINATIONS):
        return False
    return True

async def process_player(session, player_username):
    logger.info(f"📥 Processing player: {player_username}")
    archives_url = f"{CHESS_COM_API_BASE_URL}/player/{player_username}/games/archives"
    archives_json = await fetch(session, archives_url)
    if not archives_json or "archives" not in archives_json:
        logger.warning(f"⚠️  No archives found for {player_username}")
        return

    games_collected = 0
    for archive_url in reversed(archives_json["archives"]):
        games_json = await fetch(session, archive_url)
        if not games_json or "games" not in games_json:
            continue

        for game_data in games_json["games"]:
            game_id = game_data["url"].split("/")[-1]
            if is_game_processed(game_id):
                continue

            try:
                if "pgn" not in game_data:
                    mark_game_processed(game_id, player_username)
                    continue
                    
                pgn_text = game_data["pgn"]
                pgn_io = io.StringIO(pgn_text)
                game = chess.pgn.read_game(pgn_io)
                if game and tier1_quality_filter(game):
                    with open(f"{DATA_DIR}/{player_username}_{game_id}.pgn", "w") as out_pgn:
                        exporter = chess.pgn.FileExporter(out_pgn)
                        game.accept(exporter)
                    mark_game_processed(game_id, player_username)
                    games_collected += 1
                else:
                    mark_game_processed(game_id, player_username)
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                mark_game_processed(game_id, player_username)

    logger.info(f"✅ Collected {games_collected} new games for {player_username}")

async def main_async():
    os.makedirs(DATA_DIR, exist_ok=True)
    init_db()
    async with aiohttp.ClientSession() as session:
        logger.info("🚀 Starting dynamic player discovery...")
        leaderboards_json = await fetch(session, f"{CHESS_COM_API_BASE_URL}/leaderboards")
        all_players = set(HISTORICAL_LEGENDS)
        if leaderboards_json:
            for category in ["live_blitz", "live_bullet", "live_rapid"]:
                for player in leaderboards_json.get(category, [])[:TOP_N_PLAYERS]:
                    all_players.add(player["username"])
        
        titled_players_json = await fetch(session, f"{CHESS_COM_API_BASE_URL}/titled/GM")
        if titled_players_json:
            for username in titled_players_json.get("players", [])[:TOP_N_PLAYERS]:
                all_players.add(username)

        logger.info(f"🎯 Discovered {len(all_players)} unique players. Starting collection...")
        tasks = [process_player(session, player) for player in all_players]
        await asyncio.gather(*tasks)

class PGNHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pgn"):
            logger.info(f"New local PGN file detected: {event.src_path}")

if __name__ == "__main__":
    logger = setup_logging("Collector", "collector.log")
    logger.info("🚀 BugzyEngine Collector v5.2 with Two-Tier Filtering started")
    asyncio.run(main_async())
    
    # Watch for local files
    observer = Observer()
    observer.schedule(PGNHandler(), path=DATA_DIR, recursive=False)
    observer.start()
    logger.info(f"👀 Watching for new .pgn files in {DATA_DIR}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
