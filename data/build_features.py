import os
import re
import numpy as np
import pandas as pd

INPUT_PBP   = os.path.join(os.path.dirname(__file__), "pbp_raw.parquet")
INPUT_GAMES = os.path.join(os.path.dirname(__file__), "games_raw.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "game_states.parquet")


def parse_clock(clock_str: str, period: int) -> float:
    match = re.match(r"PT(\d+)M([\d.]+)S", str(clock_str))
    if not match:
        return 0.0
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    clock_s = minutes * 60 + seconds
    periods_remaining = max(0, 4 - period)
    return periods_remaining * 720 + clock_s


def build_outcome_map(games: pd.DataFrame) -> dict:
    outcomes = {}
    for game_id, group in games.groupby("GAME_ID"):
        home = group[group["MATCHUP"].str.contains(r"vs\.", na=False)]
        if len(home) == 1:
            outcomes[game_id] = 1 if home.iloc[0]["WL"] == "W" else 0
    return outcomes


def build_home_map(games: pd.DataFrame) -> dict:
    home_rows = games[games["MATCHUP"].str.contains(r"vs\.", na=False)]
    return dict(zip(home_rows["GAME_ID"], home_rows["TEAM_ABBREVIATION"]))


def process_game(game: pd.DataFrame, home_tricode: str, win: int) -> list:
    rows = []
    home_fouls = 0
    away_fouls = 0
    last_5 = []  

    for _, play in game.iterrows():
        action = play["actionType"]
        team   = play["teamTricode"]
        period = int(play["period"])

        # Track fouls
        if action == "Foul":
            if team == home_tricode:
                home_fouls += 1
            elif team != "":
                away_fouls += 1

        # Only emit rows on scoring plays
        if action not in ("Made Shot", "Free Throw"):
            continue

        score_home = play["scoreHome"]
        score_away = play["scoreAway"]
        if pd.isna(score_home) or pd.isna(score_away):
            continue
        if score_home == "" or score_away == "":
            continue

        score_home = int(score_home)
        score_away = int(score_away)
        score_diff = score_home - score_away
        time_remaining = parse_clock(play["clock"], period)

        # Momentum: track last 5 scoring events
        last_5.append(1 if team == home_tricode else -1)
        if len(last_5) > 5:
            last_5.pop(0)
        momentum = sum(last_5) / len(last_5)

        rows.append({
            "game_id":         play["GAME_ID"],
            "period":          period,
            "time_remaining_s": time_remaining,
            "score_diff":      score_diff,
            "home_fouls":      home_fouls,
            "away_fouls":      away_fouls,
            "foul_diff":       home_fouls - away_fouls,
            "momentum":        momentum,
            "win":             win,
        })

    return rows


def main():
    print("Loading data...")
    pbp   = pd.read_parquet(INPUT_PBP)
    games = pd.read_csv(INPUT_GAMES)
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)

    outcomes  = build_outcome_map(games)
    home_map  = build_home_map(games)

    print(f"  {len(pbp)} PBP rows | {len(outcomes)} game outcomes")

    print("Building features...")
    all_rows = []
    skipped  = 0

    for game_id, game in pbp.groupby("GAME_ID"):
        if game_id not in outcomes or game_id not in home_map:
            skipped += 1
            continue
        rows = process_game(game, home_map[game_id], outcomes[game_id])
        all_rows.extend(rows)

    print(f"  Skipped {skipped} games (no outcome data)")

    df = pd.DataFrame(all_rows)
    df = df[df["time_remaining_s"] > 0] 

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")
    print(f"\n{df.describe().to_string()}")


if __name__ == "__main__":
    main()
