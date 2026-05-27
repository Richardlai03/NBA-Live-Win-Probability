import os
import time
import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

INPUT_PATH      = os.path.join(os.path.dirname(__file__), "games_raw.csv")
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "pbp_raw.parquet")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "pbp_checkpoint.parquet")

SLEEP = 0.7


def fetch_game(game_id: str) -> pd.DataFrame | None:
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        df = pbp.get_data_frames()[0]
        df["GAME_ID"] = game_id
        time.sleep(SLEEP)
        return df
    except Exception as e:
        print(f"\n  ERROR {game_id}: {e}")
        time.sleep(2.0)
        return None


def fetch_all(limit: int | None = None):
    games = pd.read_csv(INPUT_PATH)
    games["GAME_ID"] = games["GAME_ID"].astype(str).str.zfill(10)  # restore leading zeros
    game_ids = games["GAME_ID"].unique()
    if limit:
        game_ids = game_ids[:limit]

    completed = set()
    all_dfs = []
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = pd.read_parquet(CHECKPOINT_PATH)
        completed = set(ckpt["GAME_ID"].unique())
        all_dfs.append(ckpt)
        print(f"Resuming — {len(completed)} games already done")

    remaining = [g for g in game_ids if g not in completed]
    total = len(remaining)
    print(f"Fetching PBP for {total} games...")

    for i, game_id in enumerate(remaining):
        df = fetch_game(game_id)
        if df is not None:
            all_dfs.append(df)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}]", end=" ")

        if (i + 1) % 200 == 0:
            pd.concat(all_dfs, ignore_index=True).to_parquet(CHECKPOINT_PATH)
            print("checkpoint saved")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows → {OUTPUT_PATH}")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    return combined


if __name__ == "__main__":
    fetch_all(limit=None)