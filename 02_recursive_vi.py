"""
02_recursive_vi.py
------------------
Implements Romer's recursive value-of-field-position estimation
(the iterative approach described in footnote 2 of his paper).

Conceptual reminder of what we're doing:
    V_i = expected net points scored by the team with the ball,
          starting from situation i, until the game ends.

The Bellman recursion in plain English:
    V[current situation] = avg over all plays of:
        (points scored before next situation) + B * V[next situation]

    where B = +1 if same team has ball next, -1 if possession flipped.

We solve this by VALUE ITERATION:
    1. Initialize all V's to 0.
    2. For each play, compute "realized value" using current V estimates.
    3. Average those realized values per situation type → new V estimates.
    4. Repeat until V's stop changing meaningfully.

Romer focused on first quarter only to avoid end-of-half / score effects.
We do the same. We use 1st-and-10 situations as our state space, indexed
by yard line (1-99 in Romer's "own yard line" convention).
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------------
# 1. Load play-by-play data
# ---------------------------------------------------------------------------
# Use multiple seasons for better sample sizes. Romer used 3 seasons; we
# can use more since we have access to more data.
SEASONS = [2020, 2021, 2022, 2023, 2024]
print(f"Downloading play-by-play data for {SEASONS}...")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"Loaded {len(pbp):,} plays")


# ---------------------------------------------------------------------------
# 2. Filter to first-quarter plays
# ---------------------------------------------------------------------------
# Romer focused on Q1 to avoid:
#   - End-of-half urgency effects
#   - Risk-aversion from score differential late in games
# This keeps the "expected points" interpretation clean.
q1 = pbp[pbp["qtr"] == 1].copy()
print(f"First-quarter plays: {len(q1):,}")


# ---------------------------------------------------------------------------
# 3. Build a "situations" dataset
# ---------------------------------------------------------------------------
# Each row will represent one situation transition:
#   - The starting situation (yard line on this play)
#   - Net points scored before reaching the next situation
#   - The next situation (yard line + which team has the ball)
#
# We focus on 1st-and-10 plays as the "anchor" state — these are the
# clean entry points to a new set of downs. Romer's V_i is defined for
# these states.

# We need to identify, for each 1st-and-10 play, the next 1st-and-10 play
# in the SAME GAME (could be same team or opponent), and the points
# scored between them.

# Sort to ensure plays are in order
q1 = q1.sort_values(["game_id", "play_id"]).reset_index(drop=True)

# Filter to 1st-and-10 (or 1st-and-goal — treat similarly)
firstdown = q1[
    (q1["down"] == 1)
    & (q1["ydstogo"].notna())
    & (q1["yardline_100"].notna())
    & (q1["posteam"].notna())
].copy()

# Convert to "own yard line" convention (Romer's): 1 = own goal, 99 = opp goal
# nfl_data_py's yardline_100 = distance to opponent end zone, so:
firstdown["own_yardline"] = 100 - firstdown["yardline_100"]
firstdown["own_yardline"] = firstdown["own_yardline"].astype(int)

print(f"1st-down anchor situations: {len(firstdown):,}")


# ---------------------------------------------------------------------------
# 4. Compute transitions: situation → (points, next situation)
# ---------------------------------------------------------------------------
# For each 1st-down situation, we need:
#   - own_yardline (the state)
#   - posteam (which team has the ball)
#   - net points scored before the NEXT 1st-down anchor
#   - next state: own_yardline of next anchor, and whether posteam is the same

# Strategy: within each game, walk through 1st-down anchors in order.
# Between anchor t and anchor t+1, sum up points scored, and figure out
# which team has the ball at anchor t+1.

# We need scoring data. nfl_data_py provides this per play.
# Key columns:
#   - posteam_score / defteam_score: cumulative scores
#   - We can compute points scored on each play from score deltas

# Easier: use total_home_score / total_away_score (cumulative) and diff them
q1_full = q1[
    [
        "game_id",
        "play_id",
        "posteam",
        "home_team",
        "away_team",
        "down",
        "ydstogo",
        "yardline_100",
        "total_home_score",
        "total_away_score",
    ]
].copy()

# Build a list of "anchor" plays (1st-and-10 or 1st-and-goal)
anchors = q1_full[
    (q1_full["down"] == 1)
    & (q1_full["yardline_100"].notna())
    & (q1_full["posteam"].notna())
].copy()
anchors["own_yardline"] = (100 - anchors["yardline_100"]).astype(int)


def build_transitions(anchors_df, plays_df):
    """
    For each consecutive pair of 1st-down anchors within a game,
    record: starting yard line, net points scored (from posteam's view)
    between them, and the next yard line + possession indicator.
    """
    transitions = []

    # Process game by game
    for game_id, game_anchors in anchors_df.groupby("game_id"):
        game_anchors = game_anchors.sort_values("play_id").reset_index(drop=True)
        game_plays = plays_df[plays_df["game_id"] == game_id].sort_values("play_id")

        # Need home/away to compute net points from posteam's perspective
        if len(game_anchors) < 2:
            continue
        home = game_anchors["home_team"].iloc[0]

        for i in range(len(game_anchors) - 1):
            cur = game_anchors.iloc[i]
            nxt = game_anchors.iloc[i + 1]

            # Net points scored between these two anchors, from CURRENT
            # offense's perspective.
            # Score deltas:
            home_pts_scored = nxt["total_home_score"] - cur["total_home_score"]
            away_pts_scored = nxt["total_away_score"] - cur["total_away_score"]

            if cur["posteam"] == home:
                net_points = home_pts_scored - away_pts_scored
            else:
                net_points = away_pts_scored - home_pts_scored

            # B = +1 if next anchor is same team, -1 if opponent
            same_team = int(nxt["posteam"] == cur["posteam"])
            B = 1 if same_team else -1

            transitions.append(
                {
                    "from_yardline": int(cur["own_yardline"]),
                    "to_yardline": int(nxt["own_yardline"]),
                    "B": B,
                    "net_points": float(net_points),
                }
            )

    return pd.DataFrame(transitions)


print("Building transition dataset (this takes a minute)...")
transitions = build_transitions(anchors, q1_full)
print(f"Built {len(transitions):,} transitions")
print(transitions.head())


# ---------------------------------------------------------------------------
# 5. Value iteration
# ---------------------------------------------------------------------------
# Now the core algorithm. For each yard line i, we compute:
#   V[i] = mean over transitions starting at i of:
#          net_points + B * V[next_yardline]
#
# Iterate until V converges.

# Yard lines run 1 to 99. We use index 0..99 (index 0 unused for clarity).
N_YARDLINES = 100
V = np.zeros(N_YARDLINES)  # initialize all values to 0

# Convert transitions to numpy arrays for speed
from_yl = transitions["from_yardline"].values
to_yl = transitions["to_yardline"].values
B_arr = transitions["B"].values
points = transitions["net_points"].values

print("\nRunning value iteration...")
MAX_ITER = 200
TOL = 1e-4

for iteration in range(MAX_ITER):
    # Compute realized value for each transition using CURRENT V estimates
    realized = points + B_arr * V[to_yl]

    # New V[i] = average realized value over transitions starting at i
    V_new = np.zeros(N_YARDLINES)
    counts = np.zeros(N_YARDLINES)

    # Accumulate sums per yard line
    np.add.at(V_new, from_yl, realized)
    np.add.at(counts, from_yl, 1)

    # Avoid divide-by-zero for yard lines with no observations
    mask = counts > 0
    V_new[mask] = V_new[mask] / counts[mask]
    V_new[~mask] = V[~mask]  # keep old value if no data

    # Check convergence
    delta = np.max(np.abs(V_new - V))
    V = V_new

    if iteration % 10 == 0 or delta < TOL:
        print(f"  Iteration {iteration:3d}: max change = {delta:.5f}")

    if delta < TOL:
        print(f"Converged after {iteration + 1} iterations.")
        break


# ---------------------------------------------------------------------------
# 6. Inspect and plot the results
# ---------------------------------------------------------------------------
# Build a clean DataFrame for analysis
V_df = pd.DataFrame(
    {
        "own_yardline": np.arange(N_YARDLINES),
        "V": V,
        "n_observations": [
            (transitions["from_yardline"] == y).sum() for y in range(N_YARDLINES)
        ],
    }
)

# Filter to yard lines with reasonable sample sizes for plotting
plot_df = V_df[(V_df["own_yardline"] >= 1) & (V_df["own_yardline"] <= 99)]
plot_df = plot_df[plot_df["n_observations"] >= 5]

print("\nSample of estimated V values:")
print(plot_df.head(20))
print("...")
print(plot_df.tail(10))

# Plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(
    plot_df["own_yardline"],
    plot_df["V"],
    linewidth=2,
    color="darkblue",
    label="Recursive V_i (full game value)",
)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Team's own yard line (1 = own goal, 99 = opp goal)")
ax.set_ylabel("V (expected net points, rest of game)")
ax.set_title(
    f"Romer-Style Value of Field Position\n"
    f"Recursive estimation, {SEASONS[0]}-{SEASONS[-1]}, Q1 only"
)
ax.legend()
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/02_recursive_vi.png", dpi=150)
plt.show()

# Save V to disk for use in next scripts
os.makedirs("output", exist_ok=True)
V_df.to_csv("output/value_function.csv", index=False)
print("\nValue function saved to output/value_function.csv")
print("Plot saved to figures/02_recursive_vi.png")

# ---------------------------------------------------------------------------
# 7. Sanity check: compare to Romer's reported values
# ---------------------------------------------------------------------------
# Romer reported (from his Figure 1 / text):
#   V at own 1   ≈ -1.6
#   V at own 15  ≈  0.0  (approx indifference point)
#   V at opp 41  ≈  2.4  (= net value of a field goal)
#   V at opp 1   ≈  5.55

print("\n--- Sanity check vs. Romer's published values ---")
checkpoints = [
    (1, "own 1", -1.6),
    (15, "own 15", 0.0),
    (50, "midfield", None),
    (59, "opp 41", 2.4),
    (99, "opp 1", 5.55),
]
for yl, label, romer_val in checkpoints:
    if yl < N_YARDLINES:
        our_val = V[yl]
        if romer_val is not None:
            print(f"  {label:12s}: ours = {our_val:6.2f}  |  Romer = {romer_val:6.2f}")
        else:
            print(f"  {label:12s}: ours = {our_val:6.2f}")
