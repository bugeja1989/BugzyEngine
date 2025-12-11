#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process1_chesscom_collector.py: Data collector for the BugzyEngine.
"""

import requests
import json
import os
import time
import chess.pgn
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import HISTORICAL_LEGENDS, CHESS_COM_API_BASE_URL

DATA_DIR = "/home/ubuntu/BugzyEngine/data/raw_pgn"

def get_leaderboard_players():
    """Fetches the top players from the leaderboards."""
    # Placeholder: Chess.com does not have a public leaderboard API endpoint.
    # We will simulate this by taking the top 50 from the GM list.
    print("Fetching leaderboard players (simulated)...")
    gms = get_titled_players("GM")
    return [player["username"] for player in gms[:50]]

def get_titled_players(title="GM"):
    """Fetches all players with a given title and filters them by ELO."""
    print(f"Fetching {title}s...")
    try:
        response = requests.get(f"{CHESS_COM_API_BASE_URL}/titled/{title}")
        response.raise_for_status()
        players = response.json().get("players", [])
        
        # Filter by ELO > 2600
        strong_players = []
        for username in players:
            try:
                stats_res = requests.get(f"{CHESS_COM_API_BASE_URL}/player/{username}/stats")
                stats_res.raise_for_status()
                stats = stats_res.json()
                # Check rapid, blitz, and bullet ratings
                if (stats.get("chess_rapid", {}).get("last", {}).get("rating", 0) > 2600 or
                    stats.get("chess_blitz", {}).get("last", {}).get("rating", 0) > 2600 or
                    stats.get("chess_bullet", {}).get("last", {}).get("rating", 0) > 2600):
                    strong_players.append({"username": username})
                time.sleep(0.5) # Rate limit
            except requests.exceptions.RequestException as e:
                print(f"Could not fetch stats for {username}: {e}")
        return strong_players

    except requests.exceptions.RequestException as e:
        print(f"Error fetching titled players for {title}: {e}")
        return []

def is_game_style_match(game, player_username):
    """Checks if a game matches the desired style (win, sacrifice, short)."""
    # 1. Win condition
    if game.headers["Result"] == "1/2-1/2" or game.headers["Result"] == "*" :
        return False # Exclude draws and ongoing games
    
    # Check if the player won
    white_player = game.headers.get("White", "").lower()
    black_player = game.headers.get("Black", "").lower()
    result = game.headers["Result"]

    player_won = False
    if result == "1-0" and player_username.lower() == white_player:
        player_won = True
    elif result == "0-1" and player_username.lower() == black_player:
        player_won = True

    if not player_won:
        return False

    # 2. Short game condition
    if len(list(game.mainline_moves())) < 40:
        return True

    # 3. Sacrifice condition (heuristic)
    board = game.board()
    for move in game.mainline_moves():
        if board.is_capture(move):
            # Simple heuristic: if a capture is made and the engine's evaluation 
            # of the position *after* the capture is worse for the capturing side, 
            # it might be a sacrifice. This is complex to do without a full engine.
            # For now, we'll focus on wins and short games.
            pass
        board.push(move)

    return True # Default to true if it's a win

def get_player_games(player_username):
    """Fetches and filters games for a specific player."""
    print(f"Fetching game archives for {player_username}...")
    try:
        response = requests.get(f"{CHESS_COM_API_BASE_URL}/player/{player_username}/games/archives")
        response.raise_for_status()
        archives = response.json().get("archives", [])
        print(f"Found {len(archives)} monthly archives for {player_username}.")

        for archive_url in archives:
            try:
                pgn_res = requests.get(archive_url + "/pgn")
                pgn_res.raise_for_status()
                pgn_text = pgn_res.text
                pgn_file = open(os.path.join(DATA_DIR, f"{player_username}_temp.pgn"), "w")
                pgn_file.write(pgn_text)
                pgn_file.close()

                saved_games = 0
                with open(os.path.join(DATA_DIR, f"{player_username}_temp.pgn")) as pgn:
                    while True:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break
                        if is_game_style_match(game, player_username):
                            # Save game to a new PGN file
                            game_id = game.headers.get("Link", "unknown").split("/")[-1]
                            with open(os.path.join(DATA_DIR, f"{player_username}_{game_id}.pgn"), "w") as out_pgn:
                                exporter = chess.pgn.FileExporter(out_pgn)
                                game.accept(exporter)
                                saved_games += 1
                os.remove(os.path.join(DATA_DIR, f"{player_username}_temp.pgn"))
                print(f"Saved {saved_games} games from archive {archive_url}")
                time.sleep(1) # Avoid rate limiting
            except requests.exceptions.RequestException as e:
                print(f"Error fetching PGN data from {archive_url}: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching game archives for {player_username}: {e}")

class PGNHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".pgn"):
            print(f"New local PGN file detected: {event.src_path}. Processing...")
            # We can add more sophisticated processing here later
            # For now, we just acknowledge it.

def main():
    """Main function for the data collector."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Start Folder Watcher ---
    event_handler = PGNHandler()
    observer = Observer()
    observer.schedule(event_handler, path=DATA_DIR, recursive=False)
    observer.start()
    print(f"Watching for new .pgn files in {DATA_DIR}")

    # --- Dynamic Player Discovery ---
    target_players = set(HISTORICAL_LEGENDS)
    print("Building target player list...")

    leaderboard_players = get_leaderboard_players()
    for player in leaderboard_players:
        target_players.add(player)

    strong_gms = get_titled_players("GM")
    for player in strong_gms:
        target_players.add(player["username"])
        
    strong_ims = get_titled_players("IM")
    for player in strong_ims:
        target_players.add(player["username"])

    print(f"Discovered {len(target_players)} unique players to target.")

    # --- Game Collection ---
    for player in list(target_players):
        get_player_games(player)

    print("Initial game collection complete. Collector will continue to watch for local files.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
