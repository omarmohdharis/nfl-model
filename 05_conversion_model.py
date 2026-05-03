"""
05_conversion_model.py
----------------------
Estimate the probability of getting a first down (or touchdown)
when going for it on 4th down.

The data challenge:
    Teams rarely go for it on 4th down (especially in the era this data
    covers — though it's increasing). 4th-down attempts are sparse,
    especially at specific yards-to-go and field positions.

Romer's solution:
    Use 3rd-down plays as a proxy. The reasoning: on 3rd-and-X, an
    offense behaves similarly to how it would on 4th-and-X — it needs
    to gain X yards. The defense behaves similarly too. So P(convert
    on 3rd-and-X) is a good estimator for P(convert on 4th-and-X).

The bias to be aware of:
    On 3rd down, an offense might choose more conservative plays
    (knowing they have a 4th down backup), while on 4th down they
    must succeed or lose possession. This MIGHT cause 4th-down
    conversion rates to be slightly higher in practice. Romer
    discusses this and concludes the bias is small. We'll combine
    3rd and 4th down plays to get a better estimate.

What we're modeling:
    P(convert) as a function of:
        - yards to go (1 to ~15)
        - field position (mostly to handle goal-line situations)

    For most of the field, field position barely matters — a team
    needing 5 yards on its own 30 has a similar conversion rate to
    a team needing 5 yards on its 60. The main exception is near
    the opponent's goal line, where the field is compressed.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import os


# ---------------------------------------------------------------------------
# 1. Load play-by-play data
# ---------------------------------------------------------------------------
SEASONS = [2020, 2021, 2022, 2023, 2024]
print(f"Loading play-by-play data for {SEASONS}...")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"Loaded {len(pbp):,} plays")


# ---------------------------------------------------------------------------
# 2. Filter to plays we'll use to estimate conversion probability
# ---------------------------------------------------------------------------
# Romer used 3rd downs only because he had to (1998-2000 had very few 4th
# down attempts). We have more 4th down attempts now, especially recent
# years. We'll use both 3rd and 4th downs to maximize our sample.
#
# What counts as "going for it"? A play where the offense ran or passed.
# Punts and field goal attempts don't count — those are kicking decisions.

attempts = pbp[
    (pbp["down"].isin([3, 4]))
    & (pbp["play_type"].isin(["pass", "run"]))  # actual offensive plays
    & (pbp["ydstogo"].notna())
    & (pbp["yardline_100"].notna())
    & (pbp["yards_gained"].notna())
].copy()

print(f"\n3rd and 4th down offensive plays: {len(attempts):,}")
print(f"  3rd downs: {(attempts['down'] == 3).sum():,}")
print(f"  4th downs: {(attempts['down'] == 4).sum():,}")


# ---------------------------------------------------------------------------
# 3. Define what counts as "converted"
# ---------------------------------------------------------------------------
# A conversion happens if EITHER:
#   - The team gained enough yards for a first down
#   - The team scored a touchdown
#
# nflfastR has a 'first_down' flag, but it's safer to recompute it because
# it can have edge cases. We use yards_gained vs ydstogo, plus check for TDs.

attempts["converted"] = (
    (attempts["yards_gained"] >= attempts["ydstogo"])
    | (attempts["touchdown"] == 1)
).astype(int)

print(f"\nOverall conversion rate: {attempts['converted'].mean():.1%}")
print(f"\nConversion rate by yards-to-go:")
yardage_buckets = (
    attempts.groupby(attempts["ydstogo"].clip(upper=20))["converted"]
    .agg(["mean", "count"])
    .reset_index()
)
print(yardage_buckets.head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# 4. Filter to a reasonable range
# ---------------------------------------------------------------------------
# We trim extreme cases:
#   - ydstogo > 15: rare (long-yardage plays after big penalties)
#   - We keep all field positions for now; we'll handle goal-line in the model

train = attempts[(attempts["ydstogo"] >= 1) & (attempts["ydstogo"] <= 15)].copy()
print(f"\nAfter filtering to 1-15 yards to go: {len(train):,} plays")


# ---------------------------------------------------------------------------
# 5. Convert to "own yard line" convention (Romer's)
# ---------------------------------------------------------------------------
train["own_yardline"] = (100 - train["yardline_100"]).astype(int)


# ---------------------------------------------------------------------------
# 6. Fit a logistic regression model
# ---------------------------------------------------------------------------
# Inputs:
#   - ydstogo: how far we need to go
#   - own_yardline: where we are on the field
#   - Interaction term: ydstogo * (near_goal_line indicator) for goal-line cases
#
# We could fit something fancier (XGBoost, GAM, etc.), but logistic regression
# gives interpretable coefficients and the relationship is fundamentally
# monotonic — more yards-to-go → harder to convert. Logistic captures this well.

# Add a goal-line indicator: this captures the special dynamics inside the 10
train["near_goal_line"] = (train["own_yardline"] >= 90).astype(int)

# Features
X_cols = ["ydstogo", "own_yardline", "near_goal_line"]
X = train[X_cols].values
y = train["converted"].values

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print(f"\nFitted logistic model coefficients:")
for col, coef in zip(X_cols, model.coef_[0]):
    print(f"  {col:18s}: {coef:+.4f}")
print(f"  Intercept:           {model.intercept_[0]:+.4f}")

print("\nInterpretation:")
print("  - ydstogo coefficient negative → more yards needed = harder")
print("  - own_yardline coefficient ≈ 0 → field position barely matters mid-field")
print("  - near_goal_line coefficient → captures goal-line compression effect")


# ---------------------------------------------------------------------------
# 7. Generate prediction surface: P(convert) for each (ydstogo, own_yardline)
# ---------------------------------------------------------------------------
# Build a grid of all reasonable combinations and predict each one.
ydstogo_range = np.arange(1, 16)
yardline_range = np.arange(1, 100)

grid = pd.DataFrame(
    [(y, yl) for y in ydstogo_range for yl in yardline_range],
    columns=["ydstogo", "own_yardline"],
)
grid["near_goal_line"] = (grid["own_yardline"] >= 90).astype(int)

grid["p_convert"] = model.predict_proba(grid[X_cols].values)[:, 1]


# ---------------------------------------------------------------------------
# 8. Compare model predictions to empirical rates as a sanity check
# ---------------------------------------------------------------------------
# At each yards-to-go value, how does the model compare to the empirical rate?
empirical = train.groupby("ydstogo")["converted"].agg(["mean", "count"]).reset_index()

# Predicted rate at midfield (own 50) for each yardstogo
midfield_pred = grid[(grid["own_yardline"] == 50)][["ydstogo", "p_convert"]]

comparison = empirical.merge(midfield_pred, on="ydstogo")
comparison.columns = ["ydstogo", "empirical_rate", "n_plays", "predicted_at_mid"]
print("\nModel vs empirical (predicted at midfield):")
print(comparison.to_string(index=False))


# ---------------------------------------------------------------------------
# 9. Plot — conversion probability curves
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: P(convert) vs yards-to-go (model overlaid on empirical)
ax = axes[0]
ax.scatter(
    empirical["ydstogo"],
    empirical["mean"],
    s=empirical["count"] * 0.05,
    alpha=0.6,
    color="steelblue",
    label="Empirical (size = # plays)",
)
ax.plot(
    midfield_pred["ydstogo"],
    midfield_pred["p_convert"],
    color="darkred",
    linewidth=2,
    label="Logistic fit (at midfield)",
)
ax.set_xlabel("Yards to go")
ax.set_ylabel("P(convert)")
ax.set_title("Conversion Probability by Yards-to-Go")
ax.set_ylim(0, 1)
ax.legend()

# Right panel: heatmap showing P(convert) across (ydstogo, yardline) space
ax = axes[1]
pivot = grid.pivot(index="ydstogo", columns="own_yardline", values="p_convert")
sns.heatmap(
    pivot,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "P(convert)"},
    ax=ax,
)
ax.set_xlabel("Own yard line (1 = own goal, 99 = opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title("Conversion Probability by Position and Distance")
ax.invert_yaxis()  # so 1 yard to go is on top

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/05_conversion_probability.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 10. Save lookup tables
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)
grid.to_csv("output/conversion_probability.csv", index=False)
print(f"\nConversion probability grid saved to output/conversion_probability.csv")
print(f"Plot saved to figures/05_conversion_probability.png")


# ---------------------------------------------------------------------------
# 11. Summary at key situations
# ---------------------------------------------------------------------------
print("\n--- P(convert) at key situations (at midfield, own 50) ---")
for ytg in [1, 2, 3, 5, 7, 10, 15]:
    p = grid[(grid["ydstogo"] == ytg) & (grid["own_yardline"] == 50)]["p_convert"].values[0]
    print(f"  4th-and-{ytg:2d}: {p:.1%}")

print("\n--- 3rd vs 4th down conversion rate (sanity check on the proxy) ---")
# Romer was worried 4th down conversions might be HIGHER than 3rd down
# because 4th down requires success. Let's check if that's true in our data.
for ytg in [1, 2, 3, 5, 10]:
    third = train[(train["down"] == 3) & (train["ydstogo"] == ytg)]["converted"]
    fourth = train[(train["down"] == 4) & (train["ydstogo"] == ytg)]["converted"]
    if len(third) >= 30 and len(fourth) >= 30:
        print(
            f"  ydstogo = {ytg:2d}: 3rd-down rate = {third.mean():.1%} ({len(third):>4d} plays)"
            f"  |  4th-down rate = {fourth.mean():.1%} ({len(fourth):>3d} plays)"
        )
