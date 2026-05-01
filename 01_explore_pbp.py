"""
01_explore_pbp.py
-----------------
First exploratory script for the 4th down decision project.

Goal: Pull 2023 NFL play-by-play data and produce a rough preview of
Romer's value-of-field-position curve. This is intentionally crude — we're
just looking at average points scored on the *current drive* by yard line,
ignoring future possessions. The full recursive V_i estimation comes next.

This script gets you familiar with the nfl_data_py data structure and
surfaces any environment/data issues before we build anything fancy.
"""

import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# 1. Pull play-by-play data
# ---------------------------------------------------------------------------
# Start with a single season to keep things fast. We can scale up later.
print("Downloading 2023 play-by-play data...")
pbp = nfl.import_pbp_data([2023], downcast=True)
print(f"Loaded {len(pbp):,} plays")
print(f"Columns available: {len(pbp.columns)}")


# ---------------------------------------------------------------------------
# 2. Inspect the structure
# ---------------------------------------------------------------------------
# Useful columns for our purposes — get familiar with these.
key_cols = [
    "game_id",
    "play_id",
    "qtr",                 # quarter (1-4, 5 = OT)
    "down",                # 1, 2, 3, 4
    "ydstogo",             # yards to first down
    "yardline_100",        # distance to opponent's end zone (1-99)
    "play_type",           # 'pass', 'run', 'punt', 'field_goal', etc.
    "drive",               # drive number within game
    "fixed_drive_result",  # 'Touchdown', 'Field goal', 'Punt', etc.
    "epa",                 # expected points added (nflfastR's pre-computed)
    "ep",                  # expected points (nflfastR's value function)
]

print("\nFirst few rows of key columns:")
print(pbp[key_cols].head(10))


# ---------------------------------------------------------------------------
# 3. Filter to first-quarter, first-and-10 plays (Romer's focus)
# ---------------------------------------------------------------------------
# Romer focused on Q1 to avoid end-of-half and risk-aversion complications.
# We filter to 1st-and-10 to get a clean "starting a new set of downs" view.
first_q = pbp[
    (pbp["qtr"] == 1)
    & (pbp["down"] == 1)
    & (pbp["ydstogo"] == 10)
    & (pbp["yardline_100"].notna())
].copy()

print(f"\nFirst-quarter, 1st-and-10 plays: {len(first_q):,}")


# ---------------------------------------------------------------------------
# 4. Compute average points scored ON THE CURRENT DRIVE by yard line
# ---------------------------------------------------------------------------
# Map drive results to net points from the perspective of the offense.
# IMPORTANT: this is a crude approximation — it ignores what happens AFTER
# the drive (kickoffs, opponent's drive, etc). The full V_i in Romer's
# paper accounts for all of that recursively. But this gives us a quick
# sanity check that the data and our logic are sensible.
drive_points_map = {
    "Touchdown": 6.984,    # Romer's TD value (accounts for ~98% XP rate)
    "Field goal": 3.0,
    "Safety": -2.0,        # safety against the offense
    "Missed field goal": 0.0,
    "Punt": 0.0,
    "Turnover": 0.0,
    "Turnover on downs": 0.0,
    "End of half": 0.0,
    "End of game": 0.0,
    "Opp touchdown": -6.984,
}

first_q["drive_points"] = first_q["fixed_drive_result"].map(drive_points_map)

# Drop anything we couldn't map (rare edge cases)
first_q = first_q.dropna(subset=["drive_points"])

# Average drive points by field position (yardline_100 = yards to opp end zone)
avg_by_yardline = (
    first_q.groupby("yardline_100")["drive_points"]
    .agg(["mean", "count"])
    .reset_index()
)

# Romer's convention: x-axis is "team's own yard line" (1 = own goal line,
# 99 = opponent's goal line). yardline_100 is the OPPOSITE — it's distance
# TO the opponent's end zone. So "own yard line" = 100 - yardline_100.
avg_by_yardline["own_yardline"] = 100 - avg_by_yardline["yardline_100"]

print("\nSample of average drive points by field position:")
print(avg_by_yardline.sort_values("own_yardline").head(15))


# ---------------------------------------------------------------------------
# 5. Plot — compare against Romer's Figure 1 shape
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Filter to yard lines with reasonable sample sizes for cleaner plot
plot_data = avg_by_yardline[avg_by_yardline["count"] >= 5]

ax.plot(
    plot_data["own_yardline"],
    plot_data["mean"],
    linewidth=2,
    color="steelblue",
    label="Avg drive points (current drive only)",
)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Team's own yard line (1 = own goal, 99 = opp goal)")
ax.set_ylabel("Average net points")
ax.set_title(
    "Crude Value-of-Field-Position Curve (2023, Q1, 1st-and-10)\n"
    "Note: ignores future possessions — full V_i comes next"
)
ax.legend()
plt.tight_layout()

# Save to figures/ directory
import os
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/01_crude_value_curve.png", dpi=150)
plt.show()

print("\nPlot saved to figures/01_crude_value_curve.png")
print("Compare the SHAPE to Romer's Figure 1 — should be roughly similar.")
print("\nNext step: implement the full recursive V_i estimation (Romer footnote 2).")
