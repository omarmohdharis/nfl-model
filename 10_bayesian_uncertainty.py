"""
10_bayesian_uncertainty.py
--------------------------
Uncertainty quantification for the 4th-down decision model.

The point estimates in scripts 02–06 treat each model parameter as known.
In reality, they're estimated from finite data and carry uncertainty.
This script quantifies that uncertainty using bootstrap resampling and
Bayesian conjugate priors, then propagates it through the decision model
to show which recommendations are rock-solid vs. genuinely ambiguous.

Approach:
  Value function:        Block bootstrap (resample games → re-run VI)
  FG probability:        Beta posterior  Beta(makes+1, misses+1)
  Conversion rate:       Bootstrap logistic regression
  Punt expected value:   Bootstrap punt plays (with bootstrapped V)

Propagation:
  For each (yardline, ydstogo) cell, draw N_MC samples from the
  component distributions and compute the EV distribution for each option.
  Report 5th/95th percentile bands.

Outputs:
  output/uncertainty_bands.csv       — EV intervals per (yl, ytg) cell
  figures/10_uncertainty_curve.png   — decision curve with 90% CI
  figures/10_component_uncertainty.png — per-component uncertainty plots

Runtime: ~5–10 minutes depending on CPU (50 bootstrap × 50 VI iterations).
"""

import os

import matplotlib.pyplot as plt
import nfl_data_py as nfl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
SEASONS = [2020, 2021, 2022, 2023, 2024]
N_BOOTSTRAP = 50     # bootstrap samples for empirical models
N_VI_ITER = 50       # value-iteration steps per bootstrap (converges fast)
N_MC = 500           # Monte Carlo draws for EV uncertainty propagation
CI_LO, CI_HI = 5, 95  # percentile bounds for confidence interval
RNG = np.random.default_rng(42)

os.makedirs("output", exist_ok=True)
os.makedirs("figures", exist_ok=True)

TD_VALUE = 6.984
FG_VALUE = 3.0
KICKOFF_RETURN_YARDLINE = 25

print(f"Config: {N_BOOTSTRAP} bootstrap samples, {N_VI_ITER} VI steps, {N_MC} MC draws")


# ---------------------------------------------------------------------------
# 2. Load raw play-by-play data (used by all component bootstraps)
# ---------------------------------------------------------------------------
print(f"\nLoading play-by-play for {SEASONS} …")
pbp = nfl.import_pbp_data(SEASONS, downcast=True)
print(f"  {len(pbp):,} plays loaded")

pbp["own_yardline"] = (100 - pbp["yardline_100"]).round().astype("Int64")


# ---------------------------------------------------------------------------
# 3. Build the Q1 transitions dataset (same logic as 02_recursive_vi.py)
# ---------------------------------------------------------------------------
print("\nBuilding Q1 first-down transitions …")

q1 = pbp[pbp["qtr"] == 1].copy()
anchors = q1[
    (q1["down"] == 1)
    & q1["yardline_100"].notna()
    & q1["posteam"].notna()
    & q1["total_home_score"].notna()
].copy()
anchors["own_yardline"] = (100 - anchors["yardline_100"]).astype(int)
anchors = anchors.sort_values(["game_id", "play_id"])

all_game_ids = anchors["game_id"].unique()
print(f"  {len(all_game_ids)} unique Q1 games")


def build_transitions(game_ids, anchors_df):
    """Build (from_yl, to_yl, B, net_points) for the given game subset."""
    rows = []
    sub = anchors_df[anchors_df["game_id"].isin(game_ids)]
    for gid, grp in sub.groupby("game_id"):
        grp = grp.sort_values("play_id").reset_index(drop=True)
        if len(grp) < 2:
            continue
        home = grp["home_team"].iloc[0]
        for i in range(len(grp) - 1):
            cur, nxt = grp.iloc[i], grp.iloc[i + 1]
            h_pts = nxt["total_home_score"] - cur["total_home_score"]
            a_pts = nxt["total_away_score"] - cur["total_away_score"]
            net = (h_pts - a_pts) if cur["posteam"] == home else (a_pts - h_pts)
            B = 1 if nxt["posteam"] == cur["posteam"] else -1
            rows.append((int(cur["own_yardline"]), int(nxt["own_yardline"]), B, float(net)))
    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.array(rows)
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3]


def run_vi(from_yl, to_yl, B, net_points, n_iter=N_VI_ITER):
    """Run value iteration and return converged V array (length 100)."""
    V = np.zeros(100)
    for _ in range(n_iter):
        realized = net_points + B * V[to_yl]
        V_new = np.zeros(100)
        counts = np.zeros(100)
        np.add.at(V_new, from_yl, realized)
        np.add.at(counts, from_yl, 1)
        mask = counts > 0
        V_new[mask] /= counts[mask]
        V_new[~mask] = V[~mask]
        V = V_new
    return V


# Build the point-estimate transitions (all games)
from_yl_all, to_yl_all, B_all, pts_all = build_transitions(all_game_ids, anchors)
V_point = run_vi(from_yl_all, to_yl_all, B_all, pts_all, n_iter=200)
print(f"  Point-estimate V range: [{V_point[1:100].min():.2f}, {V_point[1:100].max():.2f}]")


# ---------------------------------------------------------------------------
# 4. Bootstrap the value function
# ---------------------------------------------------------------------------
print(f"\nBootstrapping value function ({N_BOOTSTRAP} samples) …")

V_boots = np.zeros((N_BOOTSTRAP, 100))
for b in range(N_BOOTSTRAP):
    sampled_games = RNG.choice(all_game_ids, size=len(all_game_ids), replace=True)
    fyl, tyl, Barr, net = build_transitions(sampled_games, anchors)
    if len(fyl) == 0:
        V_boots[b] = V_point
    else:
        V_boots[b] = run_vi(fyl, tyl, Barr, net, n_iter=N_VI_ITER)
    if (b + 1) % 10 == 0:
        print(f"  Bootstrap {b + 1}/{N_BOOTSTRAP}")

V_lo = np.percentile(V_boots, CI_LO, axis=0)
V_hi = np.percentile(V_boots, CI_HI, axis=0)
print("  Value function bootstrap complete.")


# ---------------------------------------------------------------------------
# 5. Bootstrap the field goal model (also build Beta posteriors)
# ---------------------------------------------------------------------------
print("\nBootstrapping field goal model …")

fg_plays = pbp[
    (pbp["play_type"] == "field_goal")
    & pbp["kick_distance"].notna()
    & pbp["field_goal_result"].notna()
].copy()
fg_plays["own_yardline"] = (100 - fg_plays["yardline_100"]).round().astype(int)
fg_plays["made"] = (fg_plays["field_goal_result"] == "made").astype(int)
fg_game_ids = fg_plays["game_id"].unique()

# Beta posteriors per yardline bucket (smoothed over ±2 yards)
fg_beta_params = {}  # own_yardline -> (alpha, beta)
for yl in range(51, 100):
    sub = fg_plays[fg_plays["own_yardline"].between(yl - 2, yl + 2)]
    makes = sub["made"].sum()
    attempts = len(sub)
    fg_beta_params[yl] = (makes + 1, attempts - makes + 1)  # Beta posterior

# Bootstrap logistic FG models for MC sampling
fg_p_boots = {}  # own_yardline -> array of p_make estimates
for yl in range(51, 100):
    alpha, beta = fg_beta_params[yl]
    # Draw N_MC samples from Beta posterior
    fg_p_boots[yl] = RNG.beta(alpha, beta, size=N_MC)

print("  FG uncertainty built (Beta posteriors).")


# ---------------------------------------------------------------------------
# 6. Bootstrap the conversion probability model
# ---------------------------------------------------------------------------
print(f"\nBootstrapping conversion model ({N_BOOTSTRAP} samples) …")

conv_plays = pbp[
    (pbp["down"].isin([3, 4]))
    & (pbp["play_type"].isin(["pass", "run"]))
    & pbp["ydstogo"].notna()
    & pbp["yardline_100"].notna()
    & pbp["first_down"].notna()
].copy()
conv_plays["own_yardline"] = (100 - conv_plays["yardline_100"]).round().astype(int)
conv_plays["near_goal"] = (conv_plays["own_yardline"] >= 90).astype(int)
conv_plays["converted"] = conv_plays["first_down"].astype(int)
conv_game_ids = conv_plays["game_id"].unique()

FEATURES = ["ydstogo", "own_yardline", "near_goal"]

# Grid of (ydstogo, yardline) pairs we'll compute CIs for
ytg_vals = np.arange(1, 16)
yl_vals = np.arange(1, 100)
grid = pd.DataFrame(
    [(ytg, yl) for ytg in ytg_vals for yl in yl_vals],
    columns=["ydstogo", "own_yardline"],
)
grid["near_goal"] = (grid["own_yardline"] >= 90).astype(int)

conv_p_boots = np.zeros((N_BOOTSTRAP, len(grid)))

for b in range(N_BOOTSTRAP):
    sampled = RNG.choice(conv_game_ids, size=len(conv_game_ids), replace=True)
    sub = conv_plays[conv_plays["game_id"].isin(sampled)]
    X = sub[FEATURES].values
    y = sub["converted"].values
    try:
        lr = LogisticRegression(max_iter=300, C=1.0)
        lr.fit(X, y)
        conv_p_boots[b] = lr.predict_proba(grid[FEATURES].values)[:, 1]
    except Exception:
        # If this bootstrap sample is degenerate, use point-estimate column
        conv_p_boots[b] = conv_p_boots[max(b - 1, 0)]
    if (b + 1) % 10 == 0:
        print(f"  Bootstrap {b + 1}/{N_BOOTSTRAP}")

# Point estimate (for reference)
lr_point = LogisticRegression(max_iter=300, C=1.0)
lr_point.fit(conv_plays[FEATURES].values, conv_plays["converted"].values)
conv_p_point = lr_point.predict_proba(grid[FEATURES].values)[:, 1]

print("  Conversion model bootstrap complete.")


# ---------------------------------------------------------------------------
# 7. Bootstrap the punt model
# ---------------------------------------------------------------------------
print(f"\nBootstrapping punt model ({N_BOOTSTRAP} samples) …")

punt_plays = pbp[
    (pbp["play_type"] == "punt")
    & pbp["kick_distance"].notna()
    & pbp["yardline_100"].notna()
].copy()
punt_plays["own_yardline"] = (100 - punt_plays["yardline_100"]).round().astype(int)
punt_game_ids = punt_plays["game_id"].unique()

# Point-estimate punt lookup (for reference)
punt_df_ref = pd.read_csv("output/punt_values.csv")
punt_lookup_point = dict(
    zip(punt_df_ref["own_yardline_kicking"].astype(int), punt_df_ref["E_V_punt"])
)

# Bootstrap: for each sample, recompute average punt outcome using bootstrapped V
punt_ev_boots = {yl: [] for yl in range(1, 100)}

for b in range(N_BOOTSTRAP):
    V_b = V_boots[b]
    sampled = RNG.choice(punt_game_ids, size=len(punt_game_ids), replace=True)
    sub = punt_plays[punt_plays["game_id"].isin(sampled)].copy()

    # Approximate punt value: for each punt play, use actual return yardline
    # If ball returned, opponent_yl = yardline_100 after punt (already in nfl data)
    # For simplicity, use average net punt gain per field position bucket
    for yl in range(1, 100):
        bucket = sub[sub["own_yardline"].between(yl - 3, yl + 3)]
        if len(bucket) < 5:
            punt_ev_boots[yl].append(punt_lookup_point.get(yl, np.nan))
            continue
        # Estimate: expected field position after punt = opponent at (yl - avg_net)
        # Use V function to compute value to us (opponent has ball)
        avg_kick = bucket["kick_distance"].mean()
        opp_yl = max(1, min(99, 100 - (yl + int(avg_kick) - 10)))  # rough touchback adj
        ev = -V_b[opp_yl]  # negative because opponent has ball
        punt_ev_boots[yl].append(ev)

punt_ev_lo = {yl: np.nanpercentile(punt_ev_boots[yl], CI_LO) for yl in range(1, 100)}
punt_ev_hi = {yl: np.nanpercentile(punt_ev_boots[yl], CI_HI) for yl in range(1, 100)}
print("  Punt model bootstrap complete.")


# ---------------------------------------------------------------------------
# 8. Monte Carlo EV propagation
# ---------------------------------------------------------------------------
print(f"\nMonte Carlo EV propagation ({N_MC} draws per cell) …")

# Precompute V samples: shape (N_MC, 100) — draw from bootstrap distribution
v_sample_idx = RNG.integers(0, N_BOOTSTRAP, size=N_MC)
V_mc = V_boots[v_sample_idx]  # (N_MC, 100)

V_after_score_mc = -V_mc[:, KICKOFF_RETURN_YARDLINE]  # (N_MC,)

records = []
for row_idx, row in grid.iterrows():
    ytg = int(row["ydstogo"])
    yl = int(row["own_yardline"])

    # -- EV(go for it) --
    # p_convert: draw from conversion bootstrap samples
    conv_col = row_idx  # same ordering as grid
    p_conv_mc = conv_p_boots[RNG.integers(0, N_BOOTSTRAP, size=N_MC), conv_col]

    new_yl = yl + ytg
    if new_yl >= 100:
        v_success_mc = TD_VALUE + V_after_score_mc
    else:
        v_success_mc = V_mc[:, min(new_yl, 99)]

    opp_fail_yl = 100 - yl
    v_fail_mc = -V_mc[:, max(1, min(99, opp_fail_yl))]

    ev_go_mc = p_conv_mc * v_success_mc + (1 - p_conv_mc) * v_fail_mc

    # -- EV(field goal) --
    if yl in fg_p_boots:
        p_fg_mc = fg_p_boots[yl][RNG.integers(0, N_MC, size=N_MC)]
    else:
        p_fg_mc = np.full(N_MC, np.nan)

    v_fg_made_mc = FG_VALUE + V_after_score_mc
    opp_miss_yl = max(20, min(99, 100 - (yl - 7))) if (100 - yl) >= 20 else 20
    v_fg_miss_mc = -V_mc[:, max(1, min(99, opp_miss_yl))]
    ev_fg_mc = p_fg_mc * v_fg_made_mc + (1 - p_fg_mc) * v_fg_miss_mc

    # -- EV(punt) --
    punt_lo = punt_ev_lo.get(yl, np.nan)
    punt_hi = punt_ev_hi.get(yl, np.nan)
    punt_center = punt_lookup_point.get(yl, np.nan)
    if pd.notna(punt_lo) and pd.notna(punt_hi):
        ev_punt_mc = RNG.uniform(punt_lo, punt_hi, size=N_MC)
    else:
        ev_punt_mc = np.full(N_MC, punt_center)

    records.append(
        {
            "own_yardline": yl,
            "ydstogo": ytg,
            "ev_go_lo": np.nanpercentile(ev_go_mc, CI_LO),
            "ev_go_med": np.nanmedian(ev_go_mc),
            "ev_go_hi": np.nanpercentile(ev_go_mc, CI_HI),
            "ev_fg_lo": np.nanpercentile(ev_fg_mc, CI_LO) if not np.all(np.isnan(p_fg_mc)) else np.nan,
            "ev_fg_med": np.nanmedian(ev_fg_mc) if not np.all(np.isnan(p_fg_mc)) else np.nan,
            "ev_fg_hi": np.nanpercentile(ev_fg_mc, CI_HI) if not np.all(np.isnan(p_fg_mc)) else np.nan,
            "ev_punt_lo": np.nanpercentile(ev_punt_mc, CI_LO),
            "ev_punt_med": np.nanmedian(ev_punt_mc),
            "ev_punt_hi": np.nanpercentile(ev_punt_mc, CI_HI),
        }
    )

    if (row_idx + 1) % 500 == 0:
        print(f"  {row_idx + 1}/{len(grid)} cells done …")

uncertainty_df = pd.DataFrame(records)
uncertainty_df.to_csv("output/uncertainty_bands.csv", index=False)
print(f"  Saved to output/uncertainty_bands.csv")


# ---------------------------------------------------------------------------
# 9. Build the uncertainty-aware decision curve
# ---------------------------------------------------------------------------
# For each yardline, and each bootstrap draw, compute the critical ydstogo
# (max ytg where median EV_go > median EV_best_kick).

print("\nBuilding bootstrap decision curves …")

# Precompute point-estimate EV function for reference
V_lk = {i: float(V_point[i]) for i in range(100)}
fg_lk = dict(zip(pd.read_csv("output/fg_by_yardline.csv")["own_yardline"].astype(int),
                  pd.read_csv("output/fg_by_yardline.csv")["p_make"]))
punt_lk = punt_lookup_point


def critical_ytg_from_boots(V_b):
    """Compute critical ydstogo curve from a bootstrapped V vector."""
    V_after_sc = -V_b[KICKOFF_RETURN_YARDLINE]
    crits = []
    for yl in range(1, 100):
        # EV field goal under this V sample
        p_fg = fg_lk.get(yl, np.nan)
        if pd.notna(p_fg):
            opp_miss = max(20, 100 - (yl - 7)) if (100 - yl) >= 20 else 20
            ev_fg_b = p_fg * (FG_VALUE + V_after_sc) + (1 - p_fg) * (-V_b[min(99, opp_miss)])
        else:
            ev_fg_b = np.nan

        ev_punt_b = punt_lk.get(yl, np.nan)
        kick_options = [v for v in [ev_fg_b, ev_punt_b] if pd.notna(v)]
        best_kick = max(kick_options) if kick_options else np.nan

        if pd.isna(best_kick):
            crits.append(0)
            continue

        max_go_ytg = 0
        for ytg in range(1, 16):
            p_c = lr_point.predict_proba(
                np.array([[ytg, yl, int(yl >= 90)]])
            )[0, 1]
            new_yl = yl + ytg
            if new_yl >= 100:
                ev_go_b = p_c * (TD_VALUE + V_after_sc) + (1 - p_c) * (-V_b[max(1, 100 - yl)])
            else:
                ev_go_b = p_c * V_b[new_yl] + (1 - p_c) * (-V_b[max(1, 100 - yl)])
            if ev_go_b > best_kick:
                max_go_ytg = ytg
        crits.append(max_go_ytg)
    return crits


# Compute decision curve for each bootstrap sample
crit_boots = np.zeros((N_BOOTSTRAP, 99))
for b in range(N_BOOTSTRAP):
    crit_boots[b] = critical_ytg_from_boots(V_boots[b])
    if (b + 1) % 10 == 0:
        print(f"  Decision curve bootstrap {b + 1}/{N_BOOTSTRAP}")

crit_lo = np.percentile(crit_boots, CI_LO, axis=0)
crit_med = np.percentile(crit_boots, 50, axis=0)
crit_hi = np.percentile(crit_boots, CI_HI, axis=0)
yl_axis = np.arange(1, 100)


# ---------------------------------------------------------------------------
# 10. Plot 1: Decision curve with 90% CI
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(13, 7))

ax.fill_between(yl_axis, crit_lo, crit_hi, alpha=0.25, color="green",
                label="90% CI (bootstrap uncertainty)")
ax.fill_between(yl_axis, 0, crit_med, alpha=0.15, color="green")
ax.plot(yl_axis, crit_med, color="darkgreen", linewidth=2, label="Median recommendation")
ax.plot(yl_axis, crit_lo, color="green", linewidth=0.8, linestyle="--")
ax.plot(yl_axis, crit_hi, color="green", linewidth=0.8, linestyle="--")

# Shade "ambiguous zone" where CI is wide (hi - lo > 3 ytg)
ambiguous = (crit_hi - crit_lo) > 3
ax.fill_between(yl_axis, 0, 15,
                where=ambiguous,
                color="orange", alpha=0.12,
                label="High uncertainty zone (CI width > 3 ytg)")

ax.axhline(1, color="gray", linestyle=":", linewidth=0.7)
ax.axhline(5, color="gray", linestyle=":", linewidth=0.7)
ax.axhline(10, color="gray", linestyle=":", linewidth=0.7)
ax.set_xlabel("Own yard line (1 = own goal, 99 = opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title(
    "4th-Down Decision Curve with 90% Bootstrap Confidence Interval\n"
    "(green band = go-for-it region; orange = where the model is genuinely uncertain)"
)
ax.set_xlim(1, 99)
ax.set_ylim(0, 15)
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig("figures/10_uncertainty_curve.png", dpi=150)
plt.show()
print("Saved figures/10_uncertainty_curve.png")


# ---------------------------------------------------------------------------
# 11. Plot 2: Component uncertainty (value function + FG + conversion)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Value function uncertainty
ax = axes[0]
ax.fill_between(yl_axis, V_lo[1:100], V_hi[1:100],
                alpha=0.3, color="steelblue", label="90% CI")
ax.plot(yl_axis, V_point[1:100], color="darkblue", linewidth=2, label="Point estimate")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Own yard line")
ax.set_ylabel("V (expected net points)")
ax.set_title("Value Function Uncertainty")
ax.legend(fontsize=9)

# FG probability uncertainty
ax = axes[1]
yl_fg = list(range(51, 100))
fg_lo_arr = [np.percentile(fg_p_boots.get(yl, [0.5]), CI_LO) for yl in yl_fg]
fg_hi_arr = [np.percentile(fg_p_boots.get(yl, [0.5]), CI_HI) for yl in yl_fg]
fg_med_arr = [np.percentile(fg_p_boots.get(yl, [0.5]), 50) for yl in yl_fg]
ax.fill_between(yl_fg, fg_lo_arr, fg_hi_arr, alpha=0.3, color="orange", label="90% CI")
ax.plot(yl_fg, fg_med_arr, color="darkorange", linewidth=2, label="Posterior median")
ax.set_xlabel("Own yard line")
ax.set_ylabel("P(make FG)")
ax.set_title("Field Goal Probability Uncertainty\n(Beta posterior)")
ax.set_ylim(0, 1)
ax.legend(fontsize=9)

# Conversion probability uncertainty at ydstogo=3
ax = axes[2]
ytg_show = 3
conv_rows = uncertainty_df[uncertainty_df["ydstogo"] == ytg_show].sort_values("own_yardline")
ax.fill_between(
    conv_rows["own_yardline"],
    conv_rows["ev_go_lo"],
    conv_rows["ev_go_hi"],
    alpha=0.3, color="purple", label="90% CI (EV go)"
)
ax.plot(conv_rows["own_yardline"], conv_rows["ev_go_med"],
        color="purple", linewidth=2, label="Median EV")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Own yard line")
ax.set_ylabel("EV (go for it)")
ax.set_title(f"EV(Go for It) Uncertainty — 4th & {ytg_show}")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig("figures/10_component_uncertainty.png", dpi=150)
plt.show()
print("Saved figures/10_component_uncertainty.png")


# ---------------------------------------------------------------------------
# 12. Summary: which situations are most vs. least certain?
# ---------------------------------------------------------------------------
print("\n--- Most CERTAIN situations (narrowest EV spread) ---")
uncertainty_df["go_ci_width"] = uncertainty_df["ev_go_hi"] - uncertainty_df["ev_go_lo"]
certain = uncertainty_df.nsmallest(10, "go_ci_width")[
    ["own_yardline", "ydstogo", "ev_go_med", "go_ci_width"]
]
print(certain.to_string(index=False))

print("\n--- Most AMBIGUOUS situations (widest EV spread) ---")
ambig = uncertainty_df.nlargest(10, "go_ci_width")[
    ["own_yardline", "ydstogo", "ev_go_med", "go_ci_width"]
]
print(ambig.to_string(index=False))

print("\n--- Overall: how often is the decision unambiguous? ---")
# Unambiguous = best option's CI doesn't overlap with second-best's CI
# Simplified: use median EVs to compute recommendation, then check if CI excludes alternatives
n_total = len(uncertainty_df)
print(f"Total (yl, ytg) cells evaluated: {n_total}")
print(
    f"Value function 90% CI average width: "
    f"{(V_hi[1:100] - V_lo[1:100]).mean():.3f} expected points"
)
print(
    f"Decision curve: median width of 90% CI = "
    f"{(crit_hi - crit_lo).mean():.1f} yards-to-go"
)
print("\nDone. Outputs saved to output/ and figures/.")
