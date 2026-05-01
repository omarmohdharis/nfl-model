"""
03_field_goal_model.py
----------------------
Estimate field goal success probability as a function of kick distance.

Why we need this:
    On 4th down, one option is to attempt a field goal. To compare that to
    going for it or punting, we need to know how likely the kick is to
    succeed. Distance is by far the dominant factor — a 25-yard kick is
    nearly automatic; a 55-yard kick is a coin flip.

What we're building:
    A logistic regression: P(make) = 1 / (1 + exp(-(b0 + b1 * distance)))

    The output is a smooth probability curve from ~99% at short range
    down to ~30% at extreme range. We'll use this curve in script 05
    when we compute the expected value of attempting a field goal.

Modeling notes:
    - We use ALL field goal attempts in the data (not just Q1) because
      the kicking mechanics don't change by quarter. The decision-making
      analysis stays in Q1, but the SUCCESS RATE is a physical/skill
      question that's the same throughout the game.
    - Kick distance = yard line + 17 (the 7 yards to the kick spot, the
      10 yards from goal line to back of end zone where the uprights are).
    - We could add weather, kicker quality, dome vs outdoor, etc. — but
      we keep it simple for now. Distance alone explains most of it.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import os


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
SEASONS = [2020, 2021, 2022, 2023, 2024]
print(f"Loading play-by-play data for {SEASONS}...")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"Loaded {len(pbp):,} plays")


# ---------------------------------------------------------------------------
# 2. Filter to field goal attempts
# ---------------------------------------------------------------------------
# nfl_data_py marks field goal attempts with play_type == 'field_goal'.
# field_goal_result is 'made', 'missed', or 'blocked'.
# kick_distance gives the actual kick distance in yards.

fg = pbp[pbp["play_type"] == "field_goal"].copy()
print(f"\nTotal FG attempts: {len(fg):,}")

# Drop any with missing distance
fg = fg.dropna(subset=["kick_distance", "field_goal_result"])
print(f"After dropping missing data: {len(fg):,}")

# Create binary outcome: 1 if made, 0 otherwise (missed or blocked count as fail)
fg["made"] = (fg["field_goal_result"] == "made").astype(int)

# Quick look at the distribution
print(f"\nOverall make rate: {fg['made'].mean():.1%}")
print(f"Distance range: {fg['kick_distance'].min():.0f} to {fg['kick_distance'].max():.0f} yards")
print(f"\nMake rate by distance bucket:")
fg["dist_bucket"] = pd.cut(
    fg["kick_distance"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30-39", "40-49", "50-59", "60+"],
)
print(fg.groupby("dist_bucket", observed=True)["made"].agg(["mean", "count"]))


# ---------------------------------------------------------------------------
# 3. Fit a logistic regression
# ---------------------------------------------------------------------------
# Logistic regression models P(make) as a sigmoid function of distance.
# It's the right tool here: outcomes are binary, and the response curve
# (probability vs distance) really does look sigmoid-shaped.

X = fg[["kick_distance"]].values
y = fg["made"].values

model = LogisticRegression()
model.fit(X, y)

intercept = model.intercept_[0]
slope = model.coef_[0][0]
print(f"\nFitted logistic model:")
print(f"  P(make) = 1 / (1 + exp(-({intercept:.3f} + {slope:.4f} * distance)))")
print(f"  Intercept: {intercept:.3f}")
print(f"  Slope:     {slope:.4f}  (negative = longer kicks miss more)")


# ---------------------------------------------------------------------------
# 4. Generate predictions across the realistic distance range
# ---------------------------------------------------------------------------
# In the NFL, FGs can be attempted from roughly 18 yards (snap from the 1)
# up to about 65 yards (extreme range). We'll predict across this range.

distances = np.arange(18, 70).reshape(-1, 1)
probs = model.predict_proba(distances)[:, 1]  # column 1 = P(make)

# Build a lookup table we can save and use later
fg_table = pd.DataFrame(
    {"kick_distance": distances.flatten(), "p_make": probs}
)

print(f"\nPredicted make probabilities at key distances:")
for d in [20, 30, 40, 45, 50, 55, 60]:
    p = model.predict_proba([[d]])[0, 1]
    print(f"  {d}-yard kick: {p:.1%}")


# ---------------------------------------------------------------------------
# 5. Plot — empirical vs fitted
# ---------------------------------------------------------------------------
# Always sanity-check a model by overlaying it on the empirical data.
# If the curve doesn't trace through the empirical points, something's wrong.

# Compute empirical make rates per integer distance (where we have enough data)
emp = (
    fg.groupby(fg["kick_distance"].astype(int))["made"]
    .agg(["mean", "count"])
    .reset_index()
)
emp = emp[emp["count"] >= 10]  # need at least 10 attempts for a stable estimate

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Empirical: scatter, sized by sample count
ax.scatter(
    emp["kick_distance"],
    emp["mean"],
    s=emp["count"] * 0.3,
    alpha=0.5,
    color="steelblue",
    label="Empirical (size = # attempts)",
)

# Fitted curve
ax.plot(
    fg_table["kick_distance"],
    fg_table["p_make"],
    color="darkred",
    linewidth=2,
    label="Logistic regression fit",
)

ax.set_xlabel("Kick distance (yards)")
ax.set_ylabel("P(make)")
ax.set_title(
    f"Field Goal Success Probability\n{SEASONS[0]}-{SEASONS[-1]}, n = {len(fg):,} attempts"
)
ax.set_ylim(0, 1.05)
ax.legend()
plt.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/03_fg_probability.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 6. Save the lookup table
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
fg_table.to_csv("output/fg_probability.csv", index=False)
print("\nFG probability table saved to output/fg_probability.csv")
print("Plot saved to figures/03_fg_probability.png")


# ---------------------------------------------------------------------------
# 7. Convert to "yard line → P(make)" for use in the decision model
# ---------------------------------------------------------------------------
# In the decision model, we'll have a yard line (distance to opp end zone)
# and need to know P(make) for the FG from that spot.
# Kick distance = yardline_100 + 17
#   - 7 yards from line of scrimmage to where the holder catches the snap
#   - 10 yards from goal line to back of end zone (where uprights sit)

yardline_100_to_p_make = pd.DataFrame(
    {
        "yardline_100": np.arange(1, 50),  # only sensible to attempt FG inside 50
        "kick_distance": np.arange(1, 50) + 17,
    }
)
yardline_100_to_p_make["p_make"] = model.predict_proba(
    yardline_100_to_p_make[["kick_distance"]].values
)[:, 1]

# Also add "own_yardline" using Romer's convention (1 = own goal, 99 = opp goal)
yardline_100_to_p_make["own_yardline"] = 100 - yardline_100_to_p_make["yardline_100"]

yardline_100_to_p_make.to_csv("output/fg_by_yardline.csv", index=False)
print("\nFG probability by yard line saved to output/fg_by_yardline.csv")
print("\nSample of the lookup table (own yard line perspective):")
print(yardline_100_to_p_make.sort_values("own_yardline").tail(15).to_string(index=False))
