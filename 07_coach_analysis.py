"""
07_coach_analysis.py
--------------------
Compare actual coach decisions on 4th down to the model's recommendations.

Three framings we report:
    1. Agreement rate — % of 4th downs where coach matched model.
    2. EV gap — how much expected-value the coach left on the table.
    3. Aggressiveness profile — % conservative / matches / aggressive vs model.

Two scopes:
    A. All 4th downs in the first half.
    B. "Interesting" decisions only — where the EV gap between best and
       second-best option is small (i.e., genuinely contested calls).

We restrict to first half (Q1 + Q2) because the EV framework assumes
risk-neutrality over points, which breaks down late in games when score
and time create asymmetric incentives.
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------------
# 1. Load play-by-play and the component models we need
# ---------------------------------------------------------------------------
SEASONS = [2020, 2021, 2022, 2023, 2024]
print(f"Loading play-by-play for {SEASONS}...")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"Loaded {len(pbp):,} plays")

print("\nLoading component models...")
V_df = pd.read_csv("output/value_function.csv")
V_lookup = dict(zip(V_df["own_yardline"].astype(int), V_df["V"]))

fg_df = pd.read_csv("output/fg_by_yardline.csv")
fg_lookup = dict(zip(fg_df["own_yardline"].astype(int), fg_df["p_make"]))

punt_df = pd.read_csv("output/punt_values.csv")
punt_lookup = dict(zip(punt_df["own_yardline_kicking"].astype(int), punt_df["E_V_punt"]))

conv_df = pd.read_csv("output/conversion_probability.csv")
conv_lookup = {
    (int(row["ydstogo"]), int(row["own_yardline"])): row["p_convert"]
    for _, row in conv_df.iterrows()
}


# ---------------------------------------------------------------------------
# 2. Reproduce decision functions from script 06
# ---------------------------------------------------------------------------
# (These mirror script 06 exactly. We could import from it as a module,
# but for a learning project keeping them inline is more readable.)

TD_VALUE = 6.984
FG_VALUE = 3.0
KICKOFF_RETURN_YARDLINE = 25
V_AFTER_OUR_SCORE = -V_lookup.get(KICKOFF_RETURN_YARDLINE, 0.5)


def opponent_V(opponent_own_yardline):
    V_them = V_lookup.get(int(opponent_own_yardline), np.nan)
    return -V_them if pd.notna(V_them) else np.nan


def ev_go_for_it(own_yardline, ydstogo):
    p = conv_lookup.get((int(ydstogo), int(own_yardline)), np.nan)
    if pd.isna(p):
        return np.nan
    new_yl = own_yardline + ydstogo
    if new_yl >= 100:
        v_if_convert = TD_VALUE + V_AFTER_OUR_SCORE
    else:
        v_if_convert = V_lookup.get(int(new_yl), np.nan)
        if pd.isna(v_if_convert):
            return np.nan
    v_if_fail = opponent_V(100 - own_yardline)
    if pd.isna(v_if_fail):
        return np.nan
    return p * v_if_convert + (1 - p) * v_if_fail


def ev_field_goal(own_yardline):
    p = fg_lookup.get(int(own_yardline), np.nan)
    if pd.isna(p):
        return np.nan
    v_if_made = FG_VALUE + V_AFTER_OUR_SCORE
    distance_to_endzone = 100 - own_yardline
    if distance_to_endzone < 20:
        opp_own_yl = 20
    else:
        opp_own_yl = 100 - (own_yardline - 7)
    v_if_missed = opponent_V(opp_own_yl)
    if pd.isna(v_if_missed):
        return np.nan
    return p * v_if_made + (1 - p) * v_if_missed


def ev_punt(own_yardline):
    return punt_lookup.get(int(own_yardline), np.nan)


# ---------------------------------------------------------------------------
# 3. Filter to 4th down decisions in the first half
# ---------------------------------------------------------------------------
# We need to identify: down=4, qtr in [1,2], and a real decision was made
# (not a play that was blown dead before the snap, etc.).

fourth_downs = pbp[
    (pbp["down"] == 4)
    & (pbp["qtr"].isin([1, 2]))
    & (pbp["yardline_100"].notna())
    & (pbp["ydstogo"].notna())
    & (pbp["posteam"].notna())
    & (pbp["play_type"].isin(["pass", "run", "punt", "field_goal"]))
].copy()

# Convert to own_yardline convention
fourth_downs["own_yardline"] = (100 - fourth_downs["yardline_100"]).astype(int)
fourth_downs["ydstogo"] = fourth_downs["ydstogo"].astype(int)

# Restrict to ydstogo range our conversion model covers (1-15)
fourth_downs = fourth_downs[
    (fourth_downs["ydstogo"] >= 1) & (fourth_downs["ydstogo"] <= 15)
]

print(f"\n4th down decisions in first half: {len(fourth_downs):,}")


# ---------------------------------------------------------------------------
# 4. Categorize each actual decision
# ---------------------------------------------------------------------------
# Translate the nflfastR play_type into our action categories.

def actual_action(play_type):
    if play_type in ["pass", "run"]:
        return "go_for_it"
    if play_type == "field_goal":
        return "field_goal"
    if play_type == "punt":
        return "punt"
    return None


fourth_downs["actual_action"] = fourth_downs["play_type"].apply(actual_action)
print(f"\nDecision distribution (actual coach choices):")
print((fourth_downs["actual_action"].value_counts(normalize=True) * 100).round(1))


# ---------------------------------------------------------------------------
# 5. Compute model EVs for each situation
# ---------------------------------------------------------------------------
# For each 4th down in our sample, compute the EV of all three options
# under our model.

print("\nComputing model EVs for each 4th down situation...")

def compute_evs(row):
    yl = row["own_yardline"]
    ytg = row["ydstogo"]
    return pd.Series({
        "ev_go": ev_go_for_it(yl, ytg),
        "ev_fg": ev_field_goal(yl),
        "ev_pt": ev_punt(yl),
    })


evs = fourth_downs.apply(compute_evs, axis=1)
fourth_downs = pd.concat([fourth_downs, evs], axis=1)


# ---------------------------------------------------------------------------
# 6. Determine the model's recommendation and the EV of each choice
# ---------------------------------------------------------------------------
def best_action(row):
    options = {
        "go_for_it": row["ev_go"],
        "field_goal": row["ev_fg"],
        "punt": row["ev_pt"],
    }
    valid = {k: v for k, v in options.items() if pd.notna(v)}
    if not valid:
        return None
    return max(valid, key=valid.get)


def best_ev(row):
    options = [row["ev_go"], row["ev_fg"], row["ev_pt"]]
    valid = [v for v in options if pd.notna(v)]
    return max(valid) if valid else np.nan


def actual_ev(row):
    """EV of the action the coach actually took."""
    if row["actual_action"] == "go_for_it":
        return row["ev_go"]
    if row["actual_action"] == "field_goal":
        return row["ev_fg"]
    if row["actual_action"] == "punt":
        return row["ev_pt"]
    return np.nan


fourth_downs["model_action"] = fourth_downs.apply(best_action, axis=1)
fourth_downs["best_ev"] = fourth_downs.apply(best_ev, axis=1)
fourth_downs["actual_ev"] = fourth_downs.apply(actual_ev, axis=1)
fourth_downs["ev_gap"] = fourth_downs["actual_ev"] - fourth_downs["best_ev"]
# ev_gap is <= 0 by definition (you can't beat the optimal choice)


# ---------------------------------------------------------------------------
# 7. Classify each decision relative to the model
# ---------------------------------------------------------------------------
# We define an aggressiveness ordering:
#   "go_for_it" is more aggressive than both kicking options
#   "punt" and "field_goal" are both conservative, but punt is more so
#   when going for it is the optimal play

action_order = {"punt": 0, "field_goal": 1, "go_for_it": 2}


def classify(row):
    if row["actual_action"] is None or row["model_action"] is None:
        return None
    if row["actual_action"] == row["model_action"]:
        return "matches"
    actual_rank = action_order.get(row["actual_action"], 1)
    model_rank = action_order.get(row["model_action"], 1)
    if actual_rank < model_rank:
        return "more_conservative"
    return "more_aggressive"


fourth_downs["classification"] = fourth_downs.apply(classify, axis=1)


# ---------------------------------------------------------------------------
# 8. Identify the head coach for each play
# ---------------------------------------------------------------------------
# nfl_data_py has 'home_coach' and 'away_coach' on each play.
# We need to know which coach was making the decision — the coach of
# the team with the ball (posteam).

fourth_downs["coach"] = np.where(
    fourth_downs["posteam"] == fourth_downs["home_team"],
    fourth_downs["home_coach"],
    fourth_downs["away_coach"],
)

# Drop rows with missing coach
analysis = fourth_downs.dropna(subset=["coach", "model_action", "actual_action"]).copy()
print(f"\n4th downs with valid coach + model recommendation: {len(analysis):,}")


# ---------------------------------------------------------------------------
# 9. PART A — Analysis on ALL 4th downs in the first half
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART A: Analysis on all first-half 4th downs")
print("=" * 70)

MIN_DECISIONS = 50  # require at least 50 decisions per coach for stable estimates

def coach_summary(df, label=""):
    """Build a coach-level summary table."""
    summary = df.groupby("coach").agg(
        n_decisions=("ev_gap", "size"),
        agreement_rate=("classification", lambda x: (x == "matches").mean()),
        pct_more_conservative=(
            "classification", lambda x: (x == "more_conservative").mean()
        ),
        pct_more_aggressive=(
            "classification", lambda x: (x == "more_aggressive").mean()
        ),
        avg_ev_gap=("ev_gap", "mean"),
        total_ev_gap=("ev_gap", "sum"),
    ).reset_index()

    # Filter to coaches with enough decisions
    summary = summary[summary["n_decisions"] >= MIN_DECISIONS].copy()

    # Round for display
    for col in ["agreement_rate", "pct_more_conservative", "pct_more_aggressive"]:
        summary[col] = (summary[col] * 100).round(1)
    summary["avg_ev_gap"] = summary["avg_ev_gap"].round(3)
    summary["total_ev_gap"] = summary["total_ev_gap"].round(2)

    summary = summary.sort_values("avg_ev_gap", ascending=False)
    return summary


full_summary = coach_summary(analysis, "ALL")
print(f"\nCoaches with >= {MIN_DECISIONS} 4th down decisions:  {len(full_summary)}")
print(f"\nTop 10 coaches by AVG EV gap (closest to optimal):")
print(full_summary.head(10).to_string(index=False))
print(f"\nBottom 10 coaches by AVG EV gap (furthest from optimal):")
print(full_summary.tail(10).to_string(index=False))

# Most aggressive vs most conservative
print(f"\nTop 10 MOST AGGRESSIVE coaches (highest % more aggressive than model):")
print(
    full_summary.sort_values("pct_more_aggressive", ascending=False)
    .head(10)[["coach", "n_decisions", "pct_more_aggressive", "agreement_rate"]]
    .to_string(index=False)
)

print(f"\nTop 10 MOST CONSERVATIVE coaches (highest % more conservative than model):")
print(
    full_summary.sort_values("pct_more_conservative", ascending=False)
    .head(10)[["coach", "n_decisions", "pct_more_conservative", "agreement_rate"]]
    .to_string(index=False)
)


# ---------------------------------------------------------------------------
# 10. PART B — Analysis on "interesting" decisions only
# ---------------------------------------------------------------------------
# Define "interesting" as: the gap between the best and second-best option
# under the model is less than 0.5 points. These are the contested calls
# where coaching philosophy actually matters — the obvious ones (4th-and-15
# on own 20: punt) are filtered out.

def second_best_gap(row):
    evs = sorted(
        [row["ev_go"], row["ev_fg"], row["ev_pt"]],
        key=lambda x: (-np.inf if pd.isna(x) else x),
        reverse=True,
    )
    valid = [v for v in evs if pd.notna(v)]
    if len(valid) < 2:
        return np.nan
    return valid[0] - valid[1]


analysis["margin"] = analysis.apply(second_best_gap, axis=1)
interesting = analysis[analysis["margin"] < 0.5].copy()

print("\n" + "=" * 70)
print("PART B: Analysis restricted to 'interesting' decisions (margin < 0.5)")
print("=" * 70)
print(f"Interesting decisions: {len(interesting):,} of {len(analysis):,} ({len(interesting)/len(analysis)*100:.1f}%)")

interesting_summary = coach_summary(interesting, "INTERESTING")
print(f"\nCoaches with >= {MIN_DECISIONS} interesting decisions: {len(interesting_summary)}")

if len(interesting_summary) > 0:
    print(f"\nTop 10 coaches on interesting decisions (best avg EV gap):")
    print(interesting_summary.head(10).to_string(index=False))
    print(f"\nBottom 10 coaches on interesting decisions (worst avg EV gap):")
    print(interesting_summary.tail(10).to_string(index=False))


# ---------------------------------------------------------------------------
# 11. Plot 1: ranking coaches by avg EV gap
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, max(8, len(full_summary) * 0.25)))

plot_data = full_summary.sort_values("avg_ev_gap", ascending=True)
colors = [
    "darkgreen" if x > -0.05 else "darkorange" if x > -0.15 else "darkred"
    for x in plot_data["avg_ev_gap"]
]
ax.barh(plot_data["coach"], plot_data["avg_ev_gap"], color=colors, alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Average EV gap per 4th down decision (points; 0 = optimal)")
ax.set_title(
    f"NFL Head Coaches Ranked by 4th-Down Decision Quality\n"
    f"({SEASONS[0]}-{SEASONS[-1]}, first-half decisions, n >= {MIN_DECISIONS})"
)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/07_coach_ranking_ev_gap.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 12. Plot 2: aggressiveness profile (% conservative vs % aggressive)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))

# Each coach is a point in (% conservative, % aggressive) space
ax.scatter(
    full_summary["pct_more_conservative"],
    full_summary["pct_more_aggressive"],
    s=full_summary["n_decisions"] * 0.5,
    alpha=0.6,
    color="steelblue",
)

# Label notable coaches: extremes on either dimension
for _, row in full_summary.iterrows():
    if (
        row["pct_more_conservative"] > full_summary["pct_more_conservative"].quantile(0.85)
        or row["pct_more_aggressive"] > full_summary["pct_more_aggressive"].quantile(0.85)
    ):
        ax.annotate(
            row["coach"],
            (row["pct_more_conservative"], row["pct_more_aggressive"]),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

ax.set_xlabel("% of decisions where coach was MORE CONSERVATIVE than model")
ax.set_ylabel("% of decisions where coach was MORE AGGRESSIVE than model")
ax.set_title(
    "Coach Aggressiveness Profile vs Model Recommendations\n"
    "(point size = number of decisions)"
)
plt.tight_layout()
fig.savefig("figures/07_coach_aggressiveness_profile.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 13. Plot 3: where do coaches deviate? heatmap of decision class by situation
# ---------------------------------------------------------------------------
# For each (own_yardline, ydstogo) cell, what % of coaches go for it?
# Compare to whether the model says go for it.

print("\nBuilding situational heatmap...")

# Bucket field position into chunks for readability
analysis["yl_bucket"] = pd.cut(
    analysis["own_yardline"],
    bins=[0, 20, 40, 60, 80, 100],
    labels=["1-20", "21-40", "41-60", "61-80", "81-99"],
)
analysis["ytg_bucket"] = pd.cut(
    analysis["ydstogo"],
    bins=[0, 1, 3, 5, 10, 15],
    labels=["1", "2-3", "4-5", "6-10", "11-15"],
)

actual_go_rate = analysis.pivot_table(
    index="ytg_bucket",
    columns="yl_bucket",
    values="actual_action",
    aggfunc=lambda x: (x == "go_for_it").mean(),
    observed=True,
)

model_go_rate = analysis.pivot_table(
    index="ytg_bucket",
    columns="yl_bucket",
    values="model_action",
    aggfunc=lambda x: (x == "go_for_it").mean(),
    observed=True,
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    actual_go_rate * 100, annot=True, fmt=".0f", cmap="Greens",
    cbar_kws={"label": "% go-for-it"}, ax=axes[0], vmin=0, vmax=100,
)
axes[0].set_title("ACTUAL coach behavior (% who go for it)")
axes[0].set_xlabel("Own yard line")
axes[0].set_ylabel("Yards to go")
axes[0].invert_yaxis()

sns.heatmap(
    model_go_rate * 100, annot=True, fmt=".0f", cmap="Greens",
    cbar_kws={"label": "% go-for-it"}, ax=axes[1], vmin=0, vmax=100,
)
axes[1].set_title("MODEL recommendation (% go-for-it)")
axes[1].set_xlabel("Own yard line")
axes[1].set_ylabel("Yards to go")
axes[1].invert_yaxis()

plt.tight_layout()
fig.savefig("figures/07_actual_vs_model_heatmap.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 14. Save the per-coach summary tables
# ---------------------------------------------------------------------------
full_summary.to_csv("output/coach_summary_all.csv", index=False)
if len(interesting_summary) > 0:
    interesting_summary.to_csv("output/coach_summary_interesting.csv", index=False)

print(f"\nSummary tables saved to output/coach_summary_*.csv")
print(f"Plots saved to figures/07_*.png")


# ---------------------------------------------------------------------------
# 15. Quick narrative of headline findings
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("HEADLINE FINDINGS")
print("=" * 70)

avg_gap = analysis["ev_gap"].mean()
print(f"\nAcross all {len(analysis):,} first-half 4th down decisions:")
print(f"  Average EV gap: {avg_gap:.3f} points per decision")
print(f"  → Equivalent to ~{avg_gap * 6:.2f} EP per game (assuming ~6 4th downs/game)")

agreement = (analysis["classification"] == "matches").mean()
conservative = (analysis["classification"] == "more_conservative").mean()
aggressive = (analysis["classification"] == "more_aggressive").mean()
print(f"\n  League-wide decision quality:")
print(f"    Matches model:   {agreement * 100:.1f}%")
print(f"    More conservative: {conservative * 100:.1f}%")
print(f"    More aggressive:   {aggressive * 100:.1f}%")

if conservative > aggressive:
    print(f"\n  → NFL coaches are systematically MORE CONSERVATIVE than the model.")
    print(f"    This matches Romer's 2003 finding, 20+ years later.")
else:
    print(f"\n  → NFL coaches are systematically MORE AGGRESSIVE than the model.")
