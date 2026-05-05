"""
11_win_probability.py
---------------------
Converts the 4th-down decision model into win-probability terms and
computes "Expected Wins Added" (EWA) per coach.

Motivation:
  Script 07 ranks coaches by the average expected-points gap (ev_gap).
  Expected points are intuitive but abstract. Win probability (WP) is
  more meaningful: a decision that costs 0.3 EV might cost 2% win
  probability in the 1st quarter but 15% in a tie game late in the 4th.
  EWA puts every decision on the same scale — wins.

Method:
  Step 1. Train a WP model on all plays (2020–2024):
            P(posteam wins) ~ logistic(score_diff, seconds_remaining,
                                       own_yardline, timeouts)
          We use polynomial features (degree=2) to capture the non-linear
          interaction between clock and score.

  Step 2. For each 4th-down play in Q1–Q2, compute the expected WP under
          each option by simulating the resulting game state:
            WP(go for it) = p_conv * WP(new_yl, same_score)
                          + (1–p_conv) * [1 – WP(opp_yl, same_score)]
            WP(field_goal) = p_make * [1 – WP(opp_75, score+3)]
                           + (1–p_make) * [1 – WP(opp_miss_yl, same_score)]
            WP(punt)       = 1 – WP(opp_expected_yl, same_score)

  Step 3. EWA per decision = WP(optimal_action) – WP(actual_action).
          This is ≤ 0 (zero = coach chose the optimal option).
          Summed per coach → total "win probability left on the field."

  Step 4. Re-rank coaches by EWA per game (normalised for sample size).

Outputs:
  output/wp_calibration.csv         — actual vs predicted WP in bins
  output/coach_wins_added.csv       — per-coach EWA table
  figures/11_wp_calibration.png     — reliability diagram
  figures/11_coach_ewa_ranking.png  — coach ranking by EWA/game
  figures/11_ev_vs_ewa_scatter.png  — EV gap vs EWA (shows they agree, but differ)
"""

import os

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

os.makedirs("output", exist_ok=True)
os.makedirs("figures", exist_ok=True)

SEASONS = [2020, 2021, 2022, 2023, 2024]
TD_VALUE = 6.984
FG_VALUE = 3.0
KICKOFF_RETURN_YARDLINE = 25


# ---------------------------------------------------------------------------
# 1. Load play-by-play and component models
# ---------------------------------------------------------------------------
print(f"Loading play-by-play for {SEASONS} …")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"  {len(pbp):,} plays loaded")

print("\nLoading component models …")
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
    (int(r["ydstogo"]), int(r["own_yardline"])): r["p_convert"]
    for _, r in conv_df.iterrows()
}

coach_summary = pd.read_csv("output/coach_summary_all.csv")


# ---------------------------------------------------------------------------
# 2. Determine game winners (needed to train WP model)
# ---------------------------------------------------------------------------
# nflfastR's PBP has `home_score` and `away_score` at the end of each game.
# `result` = home_score - away_score (positive means home won).
# We attach the actual game winner to each play.

print("\nResolving game outcomes …")
game_outcomes = (
    pbp.groupby("game_id")
    .agg(
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_score=("home_score", "last"),
        away_score=("away_score", "last"),
    )
    .reset_index()
)
game_outcomes["home_won"] = (game_outcomes["home_score"] > game_outcomes["away_score"]).astype(int)
pbp = pbp.merge(
    game_outcomes[["game_id", "home_won"]], on="game_id", how="left"
)
pbp["posteam_won"] = np.where(
    pbp["posteam"] == pbp["home_team"],
    pbp["home_won"],
    1 - pbp["home_won"],
)
print(f"  Win labels attached. Home win rate: {game_outcomes['home_won'].mean():.1%}")


# ---------------------------------------------------------------------------
# 3. Train the win probability model
# ---------------------------------------------------------------------------
print("\nTraining win probability model …")

# Use all non-garbage-time, non-final plays with clean game state
wp_train = pbp[
    pbp["down"].isin([1, 2, 3, 4])
    & pbp["game_seconds_remaining"].notna()
    & pbp["score_differential"].notna()
    & pbp["yardline_100"].notna()
    & pbp["posteam_timeouts_remaining"].notna()
    & pbp["posteam_won"].notna()
    & (pbp["game_seconds_remaining"] > 0)
    & (pbp["game_seconds_remaining"] <= 3600)
].copy()

wp_train["own_yardline"] = (100 - wp_train["yardline_100"]).round().clip(1, 99).astype(int)
wp_train["score_diff"] = wp_train["score_differential"].clip(-35, 35)
wp_train["secs"] = wp_train["game_seconds_remaining"].clip(0, 3600)
wp_train["timeouts"] = wp_train["posteam_timeouts_remaining"].clip(0, 3)

WP_FEATURES = ["score_diff", "secs", "own_yardline", "timeouts"]
X = wp_train[WP_FEATURES].values.astype(float)
y = wp_train["posteam_won"].astype(int).values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

wp_model = Pipeline(
    [
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")),
    ]
)
wp_model.fit(X_tr, y_tr)

y_prob = wp_model.predict_proba(X_te)[:, 1]
train_acc = wp_model.score(X_te, y_te)
print(f"  WP model accuracy on held-out set: {train_acc:.3f}")

# Calibration: compare predicted WP to actual win rate in bins
fraction_of_pos, mean_predicted_value = calibration_curve(y_te, y_prob, n_bins=20)
calib_df = pd.DataFrame(
    {"predicted_wp": mean_predicted_value, "actual_win_rate": fraction_of_pos}
)
calib_df.to_csv("output/wp_calibration.csv", index=False)


def wp_predict(score_diff, secs, own_yardline, timeouts):
    """Predict P(current possessing team wins) given game state."""
    own_yardline = max(1, min(99, int(own_yardline)))
    score_diff = float(np.clip(score_diff, -35, 35))
    secs = float(np.clip(secs, 0, 3600))
    timeouts = float(np.clip(timeouts, 0, 3))
    X_in = np.array([[score_diff, secs, own_yardline, timeouts]])
    return float(wp_model.predict_proba(X_in)[0, 1])


# Sanity check
print("\n  WP sanity checks:")
for desc, args in [
    ("Tie game, 4Q 2:00, own 20", (0, 120, 20, 2)),
    ("Down 7, 4Q 2:00, own 20", (-7, 120, 20, 2)),
    ("Up 14, 4Q 2:00, own 50", (14, 120, 50, 2)),
    ("Tie game, Q1, midfield", (0, 2700, 50, 3)),
]:
    print(f"    {desc}: WP = {wp_predict(*args):.1%}")


# ---------------------------------------------------------------------------
# 4. Compute WP for each 4th-down option's expected outcomes
# ---------------------------------------------------------------------------
print("\nLoading 4th-down decisions (Q1+Q2) from raw data …")

fourth = pbp[
    (pbp["down"] == 4)
    & pbp["qtr"].isin([1, 2])
    & pbp["yardline_100"].notna()
    & pbp["ydstogo"].notna()
    & pbp["posteam"].notna()
    & pbp["play_type"].isin(["pass", "run", "punt", "field_goal"])
    & pbp["game_seconds_remaining"].notna()
    & pbp["score_differential"].notna()
    & pbp["posteam_timeouts_remaining"].notna()
].copy()

fourth["own_yardline"] = (100 - fourth["yardline_100"]).round().astype(int)
fourth["ydstogo"] = fourth["ydstogo"].astype(int)
fourth = fourth[(fourth["ydstogo"] >= 1) & (fourth["ydstogo"] <= 15)].copy()
fourth["score_diff"] = fourth["score_differential"].clip(-35, 35)
fourth["secs"] = fourth["game_seconds_remaining"]
fourth["timeouts"] = fourth["posteam_timeouts_remaining"].clip(0, 3)

# Identify head coach for each play
fourth["coach"] = np.where(
    fourth["posteam"] == fourth["home_team"],
    fourth["home_coach"],
    fourth["away_coach"],
)

# Identify actual action
def actual_action(play_type):
    if play_type in ["pass", "run"]:
        return "go_for_it"
    if play_type == "field_goal":
        return "field_goal"
    if play_type == "punt":
        return "punt"
    return None

fourth["actual_action"] = fourth["play_type"].apply(actual_action)
fourth = fourth.dropna(subset=["coach", "actual_action"]).copy()
print(f"  {len(fourth):,} 4th-down decisions with valid coach + action")


# ---------------------------------------------------------------------------
# 5. WP computation for each option
# ---------------------------------------------------------------------------
# For each play, compute WP(go), WP(fg), WP(punt) using expected resulting
# game states. A smaller secs delta is used to approximate time consumed.
# Possession flips are handled by taking (1 – WP) from the new possessing
# team's perspective.

SECS_PER_PLAY = 5  # approximate clock time per play

print("\nComputing WP for each option on each 4th-down play …")


def wp_go_for_it(row):
    yl, ytg = int(row["own_yardline"]), int(row["ydstogo"])
    p = conv_lookup.get((ytg, yl), np.nan)
    if pd.isna(p):
        return np.nan
    secs = max(0, row["secs"] - SECS_PER_PLAY)
    sd = row["score_diff"]
    to = row["timeouts"]

    # Success: still our ball, gained ydstogo
    new_yl = min(99, yl + ytg)
    if yl + ytg >= 100:
        # Scored a TD — opponent gets kickoff; we're now the defense
        wp_success = 1 - wp_predict(-(sd + 7), secs, KICKOFF_RETURN_YARDLINE, to)
    else:
        wp_success = wp_predict(sd, secs, new_yl, to)

    # Failure: opponent gets ball at our yardline
    opp_yl = max(1, 100 - yl)
    wp_fail = 1 - wp_predict(-sd, secs, opp_yl, to)

    return p * wp_success + (1 - p) * wp_fail


def wp_field_goal(row):
    yl = int(row["own_yardline"])
    p = fg_lookup.get(yl, np.nan)
    if pd.isna(p):
        return np.nan
    secs = max(0, row["secs"] - SECS_PER_PLAY)
    sd = row["score_diff"]
    to = row["timeouts"]

    # Made: opponent receives kickoff with score_diff ±3
    wp_made = 1 - wp_predict(-(sd + 3), secs, KICKOFF_RETURN_YARDLINE, to)

    # Missed: opponent gets ball at spot of kick
    if (100 - yl) < 20:
        opp_yl = 20
    else:
        opp_yl = min(99, 100 - (yl - 7))
    wp_miss = 1 - wp_predict(-sd, secs, opp_yl, to)

    return p * wp_made + (1 - p) * wp_miss


def wp_punt(row):
    yl = int(row["own_yardline"])
    secs = max(0, row["secs"] - SECS_PER_PLAY)
    sd = row["score_diff"]
    to = row["timeouts"]

    # Opponent receives punt; average landing spot ≈ own 35 for the kicker,
    # which is 100-35=65 from opponent's own goal → opponent at their own 35
    # We estimate based on the punt values: E_V_punt already baked in net pts,
    # but for WP we need the resulting field position.
    # Use a simple approximation: average punt nets ~40 yards.
    opp_yl = max(1, min(99, 100 - (yl + 38)))
    return 1 - wp_predict(-sd, secs, opp_yl, to)


print("  Computing WP_go …")
fourth["wp_go"] = fourth.apply(wp_go_for_it, axis=1)
print("  Computing WP_fg …")
fourth["wp_fg"] = fourth.apply(wp_field_goal, axis=1)
print("  Computing WP_punt …")
fourth["wp_punt"] = fourth.apply(wp_punt, axis=1)

# WP of actual action
def actual_wp(row):
    a = row["actual_action"]
    if a == "go_for_it": return row["wp_go"]
    if a == "field_goal": return row["wp_fg"]
    if a == "punt": return row["wp_punt"]
    return np.nan

fourth["wp_actual"] = fourth.apply(actual_wp, axis=1)

# Optimal WP (best available option)
fourth["wp_optimal"] = fourth[["wp_go", "wp_fg", "wp_punt"]].max(axis=1)

# EWA per decision: positive = we found a better option (coach was suboptimal)
# Here we define it as the WP left on the table (always ≤ 0 from coach's perspective;
# we negate so bigger = worse coaching on that decision).
fourth["wp_lost"] = fourth["wp_optimal"] - fourth["wp_actual"]  # ≥ 0

# EV gap (from script 07's methodology, replicated here for comparison)
def ev_go(yl, ytg):
    p = conv_lookup.get((int(ytg), int(yl)), np.nan)
    if pd.isna(p): return np.nan
    new_yl = yl + ytg
    V_suc = (TD_VALUE - V_lookup.get(KICKOFF_RETURN_YARDLINE, 0.5)) if new_yl >= 100 else V_lookup.get(int(new_yl), np.nan)
    V_fail = -V_lookup.get(max(1, 100 - yl), np.nan)
    if pd.isna(V_suc) or pd.isna(V_fail): return np.nan
    return p * V_suc + (1 - p) * V_fail

def ev_fg(yl):
    p = fg_lookup.get(int(yl), np.nan)
    if pd.isna(p): return np.nan
    v_made = FG_VALUE - V_lookup.get(KICKOFF_RETURN_YARDLINE, 0.5)
    opp = max(20, 100 - (yl - 7)) if (100 - yl) >= 20 else 20
    v_miss = -V_lookup.get(min(99, opp), np.nan)
    if pd.isna(v_miss): return np.nan
    return p * v_made + (1 - p) * v_miss

def ev_punt_fn(yl):
    return punt_lookup.get(int(yl), np.nan)

fourth["ev_go"] = fourth.apply(lambda r: ev_go(r["own_yardline"], r["ydstogo"]), axis=1)
fourth["ev_fg"] = fourth["own_yardline"].apply(ev_fg)
fourth["ev_pt"] = fourth["own_yardline"].apply(ev_punt_fn)
fourth["ev_best"] = fourth[["ev_go", "ev_fg", "ev_pt"]].max(axis=1)
fourth["ev_actual"] = fourth.apply(
    lambda r: {"go_for_it": r["ev_go"], "field_goal": r["ev_fg"], "punt": r["ev_pt"]}.get(
        r["actual_action"], np.nan
    ),
    axis=1,
)
fourth["ev_gap"] = fourth["ev_actual"] - fourth["ev_best"]


# ---------------------------------------------------------------------------
# 6. Aggregate by coach
# ---------------------------------------------------------------------------
print("\nAggregating by coach …")
MIN_DECISIONS = 50

ewa_summary = (
    fourth.dropna(subset=["coach", "wp_lost"])
    .groupby("coach")
    .agg(
        n_decisions=("wp_lost", "size"),
        total_wp_lost=("wp_lost", "sum"),
        avg_wp_lost_per_decision=("wp_lost", "mean"),
        avg_ev_gap=("ev_gap", "mean"),
        n_seasons=(
            "season",
            lambda x: x.nunique() if "season" in x.index.names or True else 1,
        ),
    )
    .reset_index()
)

# Estimate seasons coached (from n_decisions, roughly 6 fourth-down decisions/game × 8 games/half)
ewa_summary["est_games"] = (ewa_summary["n_decisions"] / 6).round(0)

# Total WP lost → "Expected Wins Added" (negative = coach left WP on table)
# Expressed as: total WP advantage ceded per season of decisions
ewa_summary["total_wp_lost_pct"] = ewa_summary["total_wp_lost"] * 100
ewa_summary["wp_lost_per_game"] = (
    ewa_summary["total_wp_lost"] / ewa_summary["est_games"]
)

ewa_summary = ewa_summary[ewa_summary["n_decisions"] >= MIN_DECISIONS].sort_values(
    "avg_wp_lost_per_decision"
)

ewa_summary.to_csv("output/coach_wins_added.csv", index=False)

print(f"\n  Coaches with >= {MIN_DECISIONS} decisions: {len(ewa_summary)}")
print("\n  Top 10 — least WP lost per decision (best decision-making):")
print(
    ewa_summary.head(10)[
        ["coach", "n_decisions", "avg_wp_lost_per_decision", "avg_ev_gap"]
    ].to_string(index=False)
)
print("\n  Bottom 10 — most WP lost per decision (worst decision-making):")
print(
    ewa_summary.tail(10)[
        ["coach", "n_decisions", "avg_wp_lost_per_decision", "avg_ev_gap"]
    ].to_string(index=False)
)


# ---------------------------------------------------------------------------
# 7. Plot 1: Coach ranking by WP lost per decision
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, max(8, len(ewa_summary) * 0.25)))

plot_data = ewa_summary.sort_values("avg_wp_lost_per_decision", ascending=False)
colors = [
    "darkgreen" if x < 0.002 else "darkorange" if x < 0.005 else "darkred"
    for x in plot_data["avg_wp_lost_per_decision"]
]
ax.barh(
    plot_data["coach"],
    plot_data["avg_wp_lost_per_decision"] * 100,
    color=colors,
    alpha=0.8,
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Average Win Probability lost per 4th-down decision (%)")
ax.set_title(
    f"NFL Head Coaches Ranked by 4th-Down Decision Quality (Win Probability)\n"
    f"({SEASONS[0]}–{SEASONS[-1]}, first-half decisions, n ≥ {MIN_DECISIONS})"
)
plt.tight_layout()
fig.savefig("figures/11_coach_ewa_ranking.png", dpi=150)
plt.show()
print("Saved figures/11_coach_ewa_ranking.png")


# ---------------------------------------------------------------------------
# 8. Plot 2: WP calibration (reliability diagram)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
ax.plot(
    calib_df["predicted_wp"],
    calib_df["actual_win_rate"],
    "o-",
    color="steelblue",
    linewidth=2,
    label="Our model",
)
ax.set_xlabel("Predicted win probability")
ax.set_ylabel("Actual win rate")
ax.set_title("Win Probability Model Calibration\n(closer to dashed line = better)")
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
fig.savefig("figures/11_wp_calibration.png", dpi=150)
plt.show()
print("Saved figures/11_wp_calibration.png")


# ---------------------------------------------------------------------------
# 9. Plot 3: EV gap vs WP lost scatter (per coach)
# ---------------------------------------------------------------------------
# These two metrics should broadly agree — if they diverge, it tells us
# something interesting about situational context.
merged = ewa_summary.merge(
    coach_summary[["coach", "avg_ev_gap"]], on="coach", how="inner"
)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(
    merged["avg_ev_gap"],
    merged["avg_wp_lost_per_decision"] * 100,
    s=merged["n_decisions"] * 0.4,
    alpha=0.6,
    color="steelblue",
)
for _, row in merged.iterrows():
    if (
        row["avg_ev_gap"] < merged["avg_ev_gap"].quantile(0.15)
        or row["avg_ev_gap"] > merged["avg_ev_gap"].quantile(0.85)
        or row["avg_wp_lost_per_decision"] > merged["avg_wp_lost_per_decision"].quantile(0.85)
    ):
        ax.annotate(
            row["coach"],
            (row["avg_ev_gap"], row["avg_wp_lost_per_decision"] * 100),
            fontsize=8,
            xytext=(5, 5),
            textcoords="offset points",
        )

ax.set_xlabel("Avg EV gap per decision (script 07 metric; 0 = optimal)")
ax.set_ylabel("Avg WP lost per decision (%)")
ax.set_title(
    "EV Gap vs Win Probability Lost per Coach\n"
    "(point size = number of decisions; coaches should cluster near origin)"
)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
fig.savefig("figures/11_ev_vs_ewa_scatter.png", dpi=150)
plt.show()
print("Saved figures/11_ev_vs_ewa_scatter.png")


# ---------------------------------------------------------------------------
# 10. Headline findings
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("HEADLINE FINDINGS")
print("=" * 65)

avg_wp_lost = fourth["wp_lost"].mean()
total_decisions = len(fourth.dropna(subset=["wp_lost"]))
print(
    f"\nAcross {total_decisions:,} first-half 4th-down decisions ({SEASONS[0]}–{SEASONS[-1]}):"
)
print(f"  Average WP lost per decision: {avg_wp_lost * 100:.3f}%")
print(
    f"  Assuming ~6 4th downs/game → ~{avg_wp_lost * 6 * 100:.2f}% WP lost per game"
)
print(
    f"  Over a 17-game season → ~{avg_wp_lost * 6 * 17:.3f} expected wins left on the table"
)

best_coach = ewa_summary.iloc[0]
worst_coach = ewa_summary.iloc[-1]
print(f"\n  Best decision-maker: {best_coach['coach']}")
print(f"    Avg WP lost: {best_coach['avg_wp_lost_per_decision'] * 100:.4f}%/decision")
print(f"\n  Most room for improvement: {worst_coach['coach']}")
print(f"    Avg WP lost: {worst_coach['avg_wp_lost_per_decision'] * 100:.4f}%/decision")

spread = (worst_coach["avg_wp_lost_per_decision"] - best_coach["avg_wp_lost_per_decision"]) * 6 * 17
print(
    f"\n  Season-level spread (best vs worst coach, 6 4th-downs/game):"
    f"  ~{spread:.2f} expected wins difference"
)

print("\nAll outputs saved to output/ and figures/.")
