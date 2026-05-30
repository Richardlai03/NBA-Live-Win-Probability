import re
from nba_api.live.nba.endpoints import boxscore, playbyplay


def get_current_game_state(game_id: str) -> dict | None:
    try:
        bs = boxscore.BoxScore(game_id=game_id)
        game = bs.get_dict()["game"]
        period = int(game["period"])
        home_score = int(game["homeTeam"]["score"])
        away_score = int(game["awayTeam"]["score"])
        game_clock = game.get("gameClock", "PT12M00.00S")
        time_remaining = _parse_clock(game_clock, period)
        score_diff = home_score - away_score
        lead_leverage = score_diff / ((time_remaining ** 0.5) + 1)

        pbp      = playbyplay.PlayByPlay(game_id=game_id)
        actions  = pbp.get_dict().get("game", {}).get("actions", [])
        last_play = actions[-1].get("description", "") if actions else ""

        return {
            "lead_leverage":    lead_leverage,
            "score_diff":       score_diff,
            "time_remaining_s": time_remaining,
            "period":           period,
            "last_play":        last_play,  
            "home_score":       home_score,
            "away_score":       away_score,
        }

    except Exception as e:
        print(f"live_feed error for {game_id}: {e}")
        return None


def _parse_clock(clock_str: str, period: int) -> float:
    match = re.match(r"PT(\d+)M([\d.]+)S", str(clock_str))
    if not match:
        return 0.0
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    clock_s = minutes * 60 + seconds
    if period <= 4:
        return (4 - period) * 720 + clock_s
    else:
        return max(0, period - 4 - 1) * 300 + clock_s


def get_live_game_ids() -> list[str]:
    try:
        from nba_api.live.nba.endpoints import scoreboard
        sb = scoreboard.ScoreBoard()
        games = sb.get_dict()["scoreboard"]["games"]
        return [
            g["gameId"] for g in games
            # 1 = scheduled, 2 = live, 3 = final
            if g["gameStatus"] == 2  
        ]
    except Exception as e:
        return []

def get_historical_game_state(game_id: str, event_index: int = -1) -> dict | None:
    try:
        from nba_api.stats.endpoints import playbyplayv3, boxscoretraditionalv3
        pbp    = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
        scoring = pbp[pbp["actionType"].isin(["Made Shot", "Free Throw"])].reset_index(drop=True)

        if len(scoring) == 0:
            return None

        row = scoring.iloc[event_index]
        period     = int(row["period"])
        home_score = int(row["scoreHome"]) if row["scoreHome"] != "" else 0
        away_score = int(row["scoreAway"]) if row["scoreAway"] != "" else 0
        score_diff = home_score - away_score
        time_remaining = _parse_clock(row["clock"], period)
        lead_leverage  = score_diff / ((time_remaining ** 0.5) + 1)

        return {
            "lead_leverage":    lead_leverage,
            "score_diff":       score_diff,
            "time_remaining_s": time_remaining,
            "period":           period,
            "last_play":        str(row["description"]),
            "home_score":       home_score,
            "away_score":       away_score,
        }
    except Exception as e:
        print(f"historical feed error for {game_id}: {e}")
        return None