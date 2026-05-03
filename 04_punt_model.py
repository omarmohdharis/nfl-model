"""
04_punt_model.py
----------------
Full distribution-based punt outcome model.

Outcome categories:
    - normal:               regular punt (fielded, fair caught, downed, OOB)
    - normal_returned_td:   receiving team returned the punt for a TD
    - touchback:            ball reaches end zone, opponent starts at 20
    - blocked:              punt was blocked at the line of scrimmage
    - blocked_returned_td:  blocked punt returned by defense for TD (very rare)
    - muffed_recovered:     receiver muffed it, kicking team recovered

Key correction over a naive approach: we account for points scored
on the return play itself (return TDs, safeties), not just the
resulting field position. This matters because punt return TDs,
while rare (~0.4% of punts), are heavily costly when they happen.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------------
# 1. Load play-by-play data and the value function
# ---------------------------------------------------------------------------
SEASONS = [2020, 2021, 2022, 2023, 2024]
print(f"Loading play-by-play data for {SEASONS}...")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"Loaded {len(pbp):,} plays")

print("\nLoading value function from script 02...")
V_df = pd.read_csv("output/value_function.csv")
V_lookup = dict(zip(V_df["own_yardline"].astype(int), V_df["V"]))


# ---------------------------------------------------------------------------
# 2. Filter to punt plays
# ---------------------------------------------------------------------------
punts = pbp[pbp["play_type"] == "punt"].copy()
punts = punts.dropna(subset=["yardline_100", "posteam"])
print(f"Total punts: {len(punts):,}")
punts["own_yardline_kicking"] = (100 - punts["yardline_100"]).astype(int)


# ---------------------------------------------------------------------------
# 3. Categorize punt outcomes (with TD subcategories)
# ---------------------------------------------------------------------------
def categorize_punt(row):
    """Assign each punt to an outcome category."""
    blocked = row.get("punt_blocked", 0) == 1
    return_td = row.get("return_touchdown", 0) == 1
    fumble_lost = row.get("fumble_lost", 0) == 1

    if blocked:
        if return_td:
            return "blocked_returned_td"
        return "blocked"

    if fumble_lost == 1 and row.get("fumble_recovery_1_team") == row.get("posteam"):
        return "muffed_recovered"

    if row.get("touchback", 0) == 1:
        return "touchback"

    if return_td:
        return "normal_returned_td"
    return "normal"


print("\nCategorizing punt outcomes...")
punts["outcome"] = punts.apply(categorize_punt, axis=1)

print("\nOverall punt outcome distribution:")
print((punts["outcome"].value_counts(normalize=True) * 100).round(2))


# ---------------------------------------------------------------------------
# 4. Compute net points scored on the punt play itself
# ---------------------------------------------------------------------------
# This catches return TDs, safeties, and any other scoring on the play.
# Using nflfastR's pre/post score columns:
#   net points to KICKING team = (posteam_score_post - posteam_score)
#                              - (defteam_score_post - defteam_score)

punts["net_points_on_play"] = (
    (punts["posteam_score_post"] - punts["posteam_score"])
    - (punts["defteam_score_post"] - punts["defteam_score"])
)

print("\nMean net points scored on punt plays by outcome:")
print(punts.groupby("outcome")["net_points_on_play"].agg(["mean", "count"]).round(2))


# ---------------------------------------------------------------------------
# 5. Look up next-play state (where the next snap happens, who has ball)
# ---------------------------------------------------------------------------
pbp_sorted = pbp.sort_values(["game_id", "play_id"]).reset_index(drop=True)
pbp_sorted["next_yardline_100"] = pbp_sorted.groupby("game_id")["yardline_100"].shift(-1)
pbp_sorted["next_posteam"] = pbp_sorted.groupby("game_id")["posteam"].shift(-1)

punts = punts.merge(
    pbp_sorted[["game_id", "play_id", "next_yardline_100", "next_posteam"]],
    on=["game_id", "play_id"],
    how="left",
)


# ---------------------------------------------------------------------------
# 6. Compute total punt value: points + value of next state
# ---------------------------------------------------------------------------
def post_punt_total_value(row):
    """
    Total value of this punt to the KICKING TEAM:
        net_points_on_play + (value of next state to kicking team)
    """
    pts = row["net_points_on_play"]
    if pd.isna(pts):
        pts = 0.0

    if pd.isna(row["next_yardline_100"]) or pd.isna(row["next_posteam"]):
        return pts

    next_yl = int(row["next_yardline_100"])
    next_own_yl = 100 - next_yl
    V_next = V_lookup.get(next_own_yl, np.nan)
    if pd.isna(V_next):
        return pts

    if row["next_posteam"] == row["posteam"]:
        return pts + V_next
    else:
        return pts - V_next


print("\nComputing total punt values (points + next-state value)...")
punts["punt_value"] = punts.apply(post_punt_total_value, axis=1)
punts_clean = punts.dropna(subset=["punt_value"]).copy()
print(f"Punts with valid total values: {len(punts_clean):,} / {len(punts):,}")


# ---------------------------------------------------------------------------
# 7. Smoothed outcome probabilities by yard line
# ---------------------------------------------------------------------------
def smoothed_outcome_probs(df, outcomes, window=5):
    results = []
    for yl in range(1, 100):
        nearby = df[
            (df["own_yardline_kicking"] >= yl - window // 2)
            & (df["own_yardline_kicking"] <= yl + window // 2)
        ]
        if len(nearby) < 10:
            row = {"own_yardline_kicking": yl, **{f"p_{o}": np.nan for o in outcomes}}
        else:
            row = {"own_yardline_kicking": yl}
            for o in outcomes:
                row[f"p_{o}"] = (nearby["outcome"] == o).mean()
        results.append(row)
    return pd.DataFrame(results)


outcomes = [
    "normal",
    "normal_returned_td",
    "touchback",
    "blocked",
    "blocked_returned_td",
    "muffed_recovered",
]

print("\nEstimating outcome probabilities by yard line...")
outcome_probs = smoothed_outcome_probs(punts_clean, outcomes, window=5)


# ---------------------------------------------------------------------------
# 8. Smoothed conditional values
# ---------------------------------------------------------------------------
def smoothed_conditional_value(df, outcomes, window=5, min_n=3):
    results = []
    for yl in range(1, 100):
        nearby = df[
            (df["own_yardline_kicking"] >= yl - window // 2)
            & (df["own_yardline_kicking"] <= yl + window // 2)
        ]
        row = {"own_yardline_kicking": yl}
        for o in outcomes:
            sub = nearby[nearby["outcome"] == o]
            if len(sub) >= min_n:
                row[f"v_{o}"] = sub["punt_value"].mean()
            else:
                row[f"v_{o}"] = np.nan
        results.append(row)
    return pd.DataFrame(results)


print("Estimating conditional expected values by yard line and outcome...")
cond_values = smoothed_conditional_value(punts_clean, outcomes, window=5)


# ---------------------------------------------------------------------------
# 9. Fill gaps with league-wide averages for rare outcomes
# ---------------------------------------------------------------------------
global_avg = punts_clean.groupby("outcome")["punt_value"].mean()
print("\nLeague-wide average punt_value by outcome:")
print(global_avg.round(2))

for o in outcomes:
    col = f"v_{o}"
    if col in cond_values.columns and o in global_avg.index:
        cond_values[col] = cond_values[col].fillna(global_avg[o])


# ---------------------------------------------------------------------------
# 10. Combine into expected punt value per yard line
# ---------------------------------------------------------------------------
probs_and_values = outcome_probs.merge(cond_values, on="own_yardline_kicking")


def expected_punt_value(row):
    total = 0.0
    total_prob = 0.0
    for o in outcomes:
        p = row.get(f"p_{o}")
        v = row.get(f"v_{o}")
        if pd.notna(p) and pd.notna(v):
            total += p * v
            total_prob += p
    if total_prob < 0.5:
        return np.nan
    return total / total_prob


probs_and_values["E_V_punt"] = probs_and_values.apply(expected_punt_value, axis=1)


# ---------------------------------------------------------------------------
# 11. Sanity check: simple average for comparison
# ---------------------------------------------------------------------------
simple_avg = (
    punts_clean.groupby("own_yardline_kicking")["punt_value"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "simple_avg_value"})
)
probs_and_values = probs_and_values.merge(
    simple_avg, on="own_yardline_kicking", how="left"
)


# ---------------------------------------------------------------------------
# 12. Plot
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(11, 9))

ax = axes[0]
mask = probs_and_values["count"].fillna(0) >= 10
plot_df = probs_and_values[mask].copy()
ax.plot(
    plot_df["own_yardline_kicking"],
    plot_df["simple_avg_value"],
    label="Simple average",
    color="gray",
    linestyle="--",
    alpha=0.7,
)
ax.plot(
    plot_df["own_yardline_kicking"],
    plot_df["E_V_punt"],
    label="Distribution-based E[V|punt]",
    color="darkblue",
    linewidth=2,
)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Kicking team's own yard line")
ax.set_ylabel("E[V] of punting")
ax.set_title("Expected Value of Punting by Field Position")
ax.legend()

ax = axes[1]
plot_df2 = outcome_probs.dropna()
ax.stackplot(
    plot_df2["own_yardline_kicking"],
    plot_df2["p_normal"],
    plot_df2["p_normal_returned_td"],
    plot_df2["p_touchback"],
    plot_df2["p_blocked"],
    plot_df2["p_blocked_returned_td"],
    plot_df2["p_muffed_recovered"],
    labels=[
        "Normal",
        "Returned for TD",
        "Touchback",
        "Blocked",
        "Blocked & returned TD",
        "Muffed (recovered)",
    ],
    alpha=0.85,
)
ax.set_xlabel("Kicking team's own yard line")
ax.set_ylabel("Outcome probability")
ax.set_title("Punt Outcome Distribution by Field Position")
ax.legend(loc="center left", fontsize=9)
ax.set_ylim(0, 1)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/04_punt_model.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 13. Save and summarize
# ---------------------------------------------------------------------------
output_cols = ["own_yardline_kicking", "E_V_punt", "simple_avg_value", "count"] + [
    f"p_{o}" for o in outcomes
]
output = probs_and_values[output_cols].copy()
os.makedirs("output", exist_ok=True)
output.to_csv("output/punt_values.csv", index=False)

print("\n--- Summary ---")
print(f"Plot saved to figures/04_punt_model.png")
print(f"Punt values saved to output/punt_values.csv")
print(f"\nE[V_punt] at key yard lines (kicking team's perspective):")
for yl in [10, 20, 30, 40, 50, 60, 70]:
    sub = output[output["own_yardline_kicking"] == yl]
    if len(sub) and pd.notna(sub.iloc[0]["E_V_punt"]):
        ev = sub.iloc[0]["E_V_punt"]
        simple = sub.iloc[0]["simple_avg_value"]
        print(f"  Own {yl:2d}:  distribution = {ev:6.2f}  |  simple avg = {simple:6.2f}")

import pandas as pd
import nfl_data_py as nfl

pbp = nfl.import_pbp_data([2023], downcast=True)
return_tds = pbp[(pbp["play_type"] == "punt") & (pbp["return_touchdown"] == 1)]

# Look at one example punt return TD and the play that follows
example = return_tds.iloc[0]
game_id = example["game_id"]
play_id = example["play_id"]

# The punt and the next 3 plays
context = pbp[
    (pbp["game_id"] == game_id) & 
    (pbp["play_id"] >= play_id) & 
    (pbp["play_id"] <= play_id + 200)
].head(4)

print(context[["play_id", "posteam", "play_type", "yardline_100", "desc"]])