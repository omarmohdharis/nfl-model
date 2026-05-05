"""
09_live_alerts.py
-----------------
Real-time 4th-down decision alerts during live NFL games.

Polls ESPN's public (unofficial) API every 30 seconds. When the current
play situation is a 4th down, it runs the component models and prints an
EV-based recommendation with a confidence rating.

Usage:
    python 09_live_alerts.py            # continuous polling (Ctrl+C to stop)
    python 09_live_alerts.py --once     # single check, then exit
    python 09_live_alerts.py --demo     # demo alert with a hardcoded situation

Requirements: pip install requests  (all other deps already in requirements.txt)

ESPN API notes:
  - Uses the public scoreboard endpoint (no API key required).
  - possessionText "BUF 25" = the ball is at the BUF 25 yard line.
    If BUF has possession → own 25 (own_yardline=25).
    If opponent has possession → at opponent's 25 (own_yardline=75).
  - Situations are de-duplicated so each (game, down_text) pair only
    triggers one alert per polling session.
"""

import argparse
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. ESPN API constants
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NFL-Model-Project/1.0)"}
DEFAULT_POLL_INTERVAL = 30  # seconds


# ---------------------------------------------------------------------------
# 2. Load the pre-computed component models from disk
# ---------------------------------------------------------------------------
print("Loading component models from output/ …")

V_df = pd.read_csv("output/value_function.csv")
V_lookup = dict(zip(V_df["own_yardline"].astype(int), V_df["V"]))

fg_df = pd.read_csv("output/fg_by_yardline.csv")
fg_lookup = dict(zip(fg_df["own_yardline"].astype(int), fg_df["p_make"]))

punt_df = pd.read_csv("output/punt_values.csv")
punt_lookup = dict(
    zip(punt_df["own_yardline_kicking"].astype(int), punt_df["E_V_punt"])
)

conv_df = pd.read_csv("output/conversion_probability.csv")
conv_lookup = {
    (int(row["ydstogo"]), int(row["own_yardline"])): row["p_convert"]
    for _, row in conv_df.iterrows()
}

print("  Models loaded.")


# ---------------------------------------------------------------------------
# 3. EV calculation functions (mirrors 06_decision_model.py exactly)
# ---------------------------------------------------------------------------
TD_VALUE = 6.984
FG_VALUE = 3.0
KICKOFF_RETURN_YARDLINE = 25
V_AFTER_OUR_SCORE = -V_lookup.get(KICKOFF_RETURN_YARDLINE, 0.5)


def _opponent_V(opp_own_yardline):
    v = V_lookup.get(int(opp_own_yardline), np.nan)
    return -v if pd.notna(v) else np.nan


def ev_go_for_it(own_yardline, ydstogo):
    p = conv_lookup.get((int(ydstogo), int(own_yardline)), np.nan)
    if pd.isna(p):
        return np.nan
    new_yl = own_yardline + ydstogo
    if new_yl >= 100:
        v_success = TD_VALUE + V_AFTER_OUR_SCORE
    else:
        v_success = V_lookup.get(int(new_yl), np.nan)
        if pd.isna(v_success):
            return np.nan
    v_fail = _opponent_V(100 - own_yardline)
    if pd.isna(v_fail):
        return np.nan
    return p * v_success + (1 - p) * v_fail


def ev_field_goal(own_yardline):
    p = fg_lookup.get(int(own_yardline), np.nan)
    if pd.isna(p):
        return np.nan
    v_made = FG_VALUE + V_AFTER_OUR_SCORE
    if (100 - own_yardline) < 20:
        opp_yl = 20
    else:
        opp_yl = 100 - (own_yardline - 7)
    v_miss = _opponent_V(opp_yl)
    if pd.isna(v_miss):
        return np.nan
    return p * v_made + (1 - p) * v_miss


def ev_punt(own_yardline):
    return punt_lookup.get(int(own_yardline), np.nan)


def compute_recommendation(own_yardline, ydstogo):
    """
    Return (evs_dict, best_action, margin_over_second_best).
    margin is the EV gap between best and second-best option.
    """
    evs = {
        "go_for_it": ev_go_for_it(own_yardline, ydstogo),
        "field_goal": ev_field_goal(own_yardline),
        "punt": ev_punt(own_yardline),
    }
    valid = {k: v for k, v in evs.items() if pd.notna(v)}
    if not valid:
        return evs, None, np.nan

    sorted_evs = sorted(valid.values(), reverse=True)
    best_action = max(valid, key=valid.get)
    margin = sorted_evs[0] - sorted_evs[1] if len(sorted_evs) >= 2 else np.nan
    return evs, best_action, margin


# ---------------------------------------------------------------------------
# 4. ESPN API helpers
# ---------------------------------------------------------------------------
def fetch_live_competitions():
    """
    Hit the ESPN scoreboard endpoint and return all in-progress competition
    dicts. Returns [] on error or when no games are live.
    """
    try:
        resp = requests.get(ESPN_SCOREBOARD_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"[{_ts()}] ESPN API error: {exc}")
        return []

    live = []
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            state = comp.get("status", {}).get("type", {}).get("state", "")
            if state == "in":
                live.append(comp)
    return live


def parse_fourth_down(competition):
    """
    Extract a 4th-down situation from an ESPN competition dict.

    Returns (own_yardline, ydstogo, context_dict) or None if the current
    play is not a valid 4th down.

    ESPN possessionText examples:
      "BUF 25"  → ball at the BUF 25-yard line
      "KC 40"   → ball at the KC 40-yard line
    """
    situation = competition.get("situation", {})
    if not situation:
        return None

    down = situation.get("down")
    if down != 4:
        return None

    ydstogo = situation.get("distance")
    if not ydstogo or not (1 <= int(ydstogo) <= 15):
        return None
    ydstogo = int(ydstogo)

    # -- Determine possession (team abbreviation) --
    possession_id = str(situation.get("possession", ""))
    competitors = competition.get("competitors", [])
    possession_abbr = None
    for comp in competitors:
        if str(comp.get("id", "")) == possession_id:
            possession_abbr = comp.get("team", {}).get("abbreviation", "")
            break

    # -- Convert field position to own_yardline --
    # Try possessionText first ("BUF 25"), fall back to downDistanceText
    own_yardline = _parse_yardline(
        situation.get("possessionText", ""),
        situation.get("downDistanceText", ""),
        possession_abbr,
    )
    if own_yardline is None:
        return None

    own_yardline = max(1, min(99, own_yardline))

    # -- Build context dict for display --
    status = competition.get("status", {})
    team_info = {
        c.get("homeAway", ""): {
            "abbr": c.get("team", {}).get("abbreviation", "?"),
            "name": c.get("team", {}).get("shortDisplayName", "?"),
            "score": c.get("score", "0"),
        }
        for c in competitors
    }
    possessing_name = ""
    if possession_abbr:
        for comp in competitors:
            if comp.get("team", {}).get("abbreviation", "") == possession_abbr:
                possessing_name = comp.get("team", {}).get(
                    "shortDisplayName", possession_abbr
                )

    home = team_info.get("home", {})
    away = team_info.get("away", {})
    game_label = (
        f"{away.get('abbr','?')} {away.get('score','0')} @ "
        f"{home.get('abbr','?')} {home.get('score','0')}"
    )

    context = {
        "quarter": status.get("period", "?"),
        "clock": status.get("displayClock", "?"),
        "game_label": game_label,
        "possessing_team": possessing_name or possession_abbr or "?",
        "down_text": situation.get("downDistanceText", f"4th & {ydstogo}"),
        "possession_text": situation.get("possessionText", ""),
    }

    return own_yardline, ydstogo, context


def _parse_yardline(possession_text, down_text, possession_abbr):
    """
    Convert ESPN field-position strings to our own_yardline convention.

    own_yardline = 1 (own goal line) … 99 (opponent's 1-yard line).
    """
    # Try "BUF 25" form
    m = re.match(r"^([A-Z]+)\s+(\d+)$", possession_text.strip())
    if m and possession_abbr:
        field_team = m.group(1)
        yardline_num = int(m.group(2))
        if field_team == possession_abbr.upper():
            return yardline_num          # own territory
        else:
            return 100 - yardline_num   # opponent's territory

    # Fallback: parse "at BUF 25" from downDistanceText
    m2 = re.search(r"at\s+([A-Z]+)\s+(\d+)", down_text)
    if m2 and possession_abbr:
        field_team = m2.group(1)
        yardline_num = int(m2.group(2))
        if field_team == possession_abbr.upper():
            return yardline_num
        else:
            return 100 - yardline_num

    return None


# ---------------------------------------------------------------------------
# 5. Alert display
# ---------------------------------------------------------------------------
OPTION_LABELS = {
    "go_for_it": "Go for it",
    "field_goal": "Field goal",
    "punt": "Punt",
}


def print_alert(own_yardline, ydstogo, context, evs, best_action, margin):
    sep = "=" * 65
    thin = "-" * 65
    print(f"\n{sep}")
    print(
        f"  LIVE 4TH DOWN  |  Q{context['quarter']} {context['clock']}  |  "
        f"{context['game_label']}"
    )
    print(f"  {context['down_text']}  (own_yardline = {own_yardline})")
    print(thin)
    print(f"  {'Option':<22} {'EV':>8}")
    print(thin)
    for key in ["go_for_it", "field_goal", "punt"]:
        ev = evs[key]
        label = OPTION_LABELS[key]
        marker = "  ← RECOMMENDED" if key == best_action else ""
        if pd.notna(ev):
            print(f"  {label:<22} {ev:>+8.3f}{marker}")
        else:
            print(f"  {label:<22} {'N/A':>8}  (out of range)")
    print(thin)
    if pd.notna(margin):
        if margin > 0.5:
            confidence = "CLEAR"
        elif margin > 0.1:
            confidence = "CLOSE CALL"
        else:
            confidence = "COIN FLIP"
        print(f"  Margin over 2nd-best: {margin:.3f} pts  [{confidence}]")
    print(sep)


# ---------------------------------------------------------------------------
# 6. Run modes
# ---------------------------------------------------------------------------
def _ts():
    return datetime.now().strftime("%H:%M:%S")


def run_demo():
    """Print a demo alert using a hardcoded situation (no live game needed)."""
    own_yardline, ydstogo = 40, 3
    evs, best_action, margin = compute_recommendation(own_yardline, ydstogo)
    context = {
        "quarter": 2,
        "clock": "8:23",
        "game_label": "BUF 7 @ KC 10",
        "possessing_team": "Bills",
        "down_text": f"4th & {ydstogo} at BUF {own_yardline}",
        "possession_text": f"BUF {own_yardline}",
    }
    print("\n[DEMO — 4th & 3 at own 40]")
    print_alert(own_yardline, ydstogo, context, evs, best_action, margin)


def run_once():
    """Check once for live 4th-down situations and exit."""
    print(f"[{_ts()}] Checking ESPN for live NFL games …")
    games = fetch_live_competitions()
    if not games:
        print("  No games currently in progress.")
        return
    print(f"  {len(games)} game(s) in progress.")
    found = 0
    for comp in games:
        result = parse_fourth_down(comp)
        if result:
            own_yardline, ydstogo, context = result
            evs, best_action, margin = compute_recommendation(own_yardline, ydstogo)
            print_alert(own_yardline, ydstogo, context, evs, best_action, margin)
            found += 1
    if found == 0:
        print("  No 4th-down situations detected right now.")


def run_poll_loop(interval=DEFAULT_POLL_INTERVAL):
    """
    Poll ESPN continuously. Only alert once per (game, down_text) pair so
    the same play doesn't fire repeatedly while the game clock is stopped.
    """
    seen = set()
    print(f"Polling every {interval}s. Press Ctrl+C to stop.\n")
    try:
        while True:
            games = fetch_live_competitions()
            if not games:
                print(f"[{_ts()}] No live games — waiting …")
            else:
                any_fourth = False
                for comp in games:
                    result = parse_fourth_down(comp)
                    if not result:
                        continue
                    own_yardline, ydstogo, context = result
                    key = (context["game_label"], context["down_text"])
                    if key in seen:
                        continue
                    seen.add(key)
                    any_fourth = True
                    evs, best_action, margin = compute_recommendation(
                        own_yardline, ydstogo
                    )
                    print_alert(
                        own_yardline, ydstogo, context, evs, best_action, margin
                    )
                if not any_fourth:
                    print(f"[{_ts()}] {len(games)} game(s) live — no 4th downs right now.")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live NFL 4th-down decision alerts via ESPN API"
    )
    parser.add_argument(
        "--once", action="store_true", help="Check once and exit"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Print demo alert (no live game needed)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.once:
        run_once()
    else:
        run_poll_loop(interval=args.interval)
