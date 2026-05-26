import os
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

SEASONS = [
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26",
]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "games_raw.csv")


def fetch_season(season: str) -> pd.DataFrame:
    print(f"  Fetching {season}...", end=" ", flush=True)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
    )
    df = finder.get_data_frames()[0]
    print(f"{len(df)} rows")
    time.sleep(0.8)
    return df


def fetch_all_seasons() -> pd.DataFrame:
    all_dfs = []
    for season in SEASONS:
        df = fetch_season(season)
        df["SEASON"] = season
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(combined)} rows → {OUTPUT_PATH}")
    print(f"Unique game IDs: {combined['GAME_ID'].nunique()}")
    return combined


if __name__ == "__main__":
    fetch_all_seasons()
