"""
06_decision_model.py
--------------------
The 4th down decision model. Combines all four component models:
    - V_i (value function from script 02)
    - P(FG make) (from script 03)
    - E[V_punt] (from script 04)
    - P(convert) (from script 05)

For any 4th down situation (yards-to-go, field position), computes:
    - EV of punting
    - EV of attempting a field goal
    - EV of going for it
    - The recommended decision (highest EV)

Then produces Romer's signature Figure 5: at each field position,
what's the maximum yards-to-go where going for it beats kicking?

Conceptual structure of each EV calculation:

    EV(punt)    = E[V_punt | yard line]   (already pre-computed)

    EV(FG)      = P(make) * (3 + V_after_kickoff)
                + (1 - P(make)) * V_after_miss

    EV(go for it) = P(convert) * V_after_success
                  + (1 - P(convert)) * V_after_failure

    where the V_after_* terms come from the V_i lookup,
    sign-flipped if the opponent ends up with the ball.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------------------------
# 1. Load all the component models
# ---------------------------------------------------------------------------
print("Loading component models...")

# Value function: V[own_yardline] -> expected net points
V_df = pd.read_csv("output/value_function.csv")
V_lookup = dict(zip(V_df["own_yardline"].astype(int), V_df["V"]))

# Field goal probability: P(make) at each yard line (for FG attempts)
fg_df = pd.read_csv("output/fg_by_yardline.csv")
fg_lookup = dict(zip(fg_df["own_yardline"].astype(int), fg_df["p_make"]))

# Punt expected value: E[V_punt] at each yard line
punt_df = pd.read_csv("output/punt_values.csv")
punt_lookup = dict(zip(punt_df["own_yardline_kicking"].astype(int), punt_df["E_V_punt"]))

# Conversion probability: P(convert) at (yards-to-go, yard line)
conv_df = pd.read_csv("output/conversion_probability.csv")
conv_lookup = {
    (int(row["ydstogo"]), int(row["own_yardline"])): row["p_convert"]
    for _, row in conv_df.iterrows()
}

print(f"  V function:              {len(V_lookup)} yard lines")
print(f"  FG probabilities:        {len(fg_lookup)} yard lines")
print(f"  Punt values:             {len(punt_lookup)} yard lines")
print(f"  Conversion probabilities: {len(conv_lookup):,} (ydstogo, yardline) pairs")


# ---------------------------------------------------------------------------
# 2. Constants we'll use
# ---------------------------------------------------------------------------
# Touchdown value (with the standard ~98% extra point success folded in)
TD_VALUE = 6.984

# Field goal value (the points themselves)
FG_VALUE = 3.0

# After a score, the scoring team kicks off. Average return puts the
# receiving team around their own 25-30. We need V at that position
# from THEIR perspective. From the SCORING team's perspective, this is
# negative (opponent has the ball).
KICKOFF_RETURN_YARDLINE = 25  # opponent's own yard line after average kickoff
V_AFTER_OUR_SCORE = -V_lookup.get(KICKOFF_RETURN_YARDLINE, 0.5)
print(f"\nValue to scoring team after kickoff: {V_AFTER_OUR_SCORE:.2f}")
print("  (negative because opponent receives the kick)")


# ---------------------------------------------------------------------------
# 3. Helper: get V from opponent's perspective
# ---------------------------------------------------------------------------
def opponent_V(opponent_own_yardline):
    """
    If opponent has the ball at THEIR own yard line Y, what's the value
    to US? It's -V[Y] (sign-flipped).
    """
    V_them = V_lookup.get(int(opponent_own_yardline), np.nan)
    if pd.isna(V_them):
        return np.nan
    return -V_them


# ---------------------------------------------------------------------------
# 4. EV of going for it
# ---------------------------------------------------------------------------
def ev_go_for_it(own_yardline, ydstogo):
    """
    Expected value of going for it on 4th down.

    If we convert:
        - We gain (at least) ydstogo yards. New 1st down at own_yardline + ydstogo.
        - If we score a TD on the play, value = TD_VALUE + V_after_our_score.
        - Otherwise, value = V[new yard line].

    If we fail:
        - Opponent gets the ball at our current yard line.
        - From our perspective: -V[their yard line] = -V[100 - own_yardline]

    For simplicity, we treat conversions as gaining exactly ydstogo yards
    (a slight underestimate — actual gains are often longer). This matches
    Romer's approach and keeps the model interpretable.
    """
    p_convert = conv_lookup.get((int(ydstogo), int(own_yardline)), np.nan)
    if pd.isna(p_convert):
        return np.nan

    # New yard line if we convert
    new_yardline = own_yardline + ydstogo

    # Value if we convert
    if new_yardline >= 100:
        # We scored a touchdown
        v_if_convert = TD_VALUE + V_AFTER_OUR_SCORE
    else:
        # We have a new 1st down
        v_if_convert = V_lookup.get(int(new_yardline), np.nan)
        if pd.isna(v_if_convert):
            return np.nan

    # Value if we fail: opponent gets ball at our position
    # Their own yard line = 100 - our own yard line
    opp_own_yardline = 100 - own_yardline
    v_if_fail = opponent_V(opp_own_yardline)
    if pd.isna(v_if_fail):
        return np.nan

    return p_convert * v_if_convert + (1 - p_convert) * v_if_fail


# ---------------------------------------------------------------------------
# 5. EV of attempting a field goal
# ---------------------------------------------------------------------------
def ev_field_goal(own_yardline):
    """
    Expected value of attempting a field goal.

    If we make it:
        - 3 points + V after our kickoff (= -V[opponent at own 25])

    If we miss:
        - Opponent gets the ball at the SPOT OF THE KICK, which is 7
          yards behind the line of scrimmage (kick is from 7 yards back).
        - Exception: if attempted from inside the 20, missed FG gives
          the ball to the opponent at the 20 (per NFL rules).
    """
    p_make = fg_lookup.get(int(own_yardline), np.nan)
    if pd.isna(p_make):
        return np.nan

    # If made
    v_if_made = FG_VALUE + V_AFTER_OUR_SCORE

    # If missed
    distance_to_endzone = 100 - own_yardline  # in yardline_100 terms
    if distance_to_endzone < 20:
        # Inside the 20 — opponent gets ball at the 20
        opp_own_yardline = 20
    else:
        # Outside the 20 — opponent gets ball at the spot of the kick
        # Spot of kick = 7 yards behind line of scrimmage
        # Their own yard line = 100 - (our_yardline - 7)
        opp_own_yardline = 100 - (own_yardline - 7)

    v_if_missed = opponent_V(opp_own_yardline)
    if pd.isna(v_if_missed):
        return np.nan

    return p_make * v_if_made + (1 - p_make) * v_if_missed


# ---------------------------------------------------------------------------
# 6. EV of punting (just look up from our pre-computed table)
# ---------------------------------------------------------------------------
def ev_punt(own_yardline):
    return punt_lookup.get(int(own_yardline), np.nan)


# ---------------------------------------------------------------------------
# 7. The decision function
# ---------------------------------------------------------------------------
def fourth_down_decision(own_yardline, ydstogo):
    """
    Compute the EV of each option and return the recommendation.

    Returns a dict with EVs for each option and the recommended action.
    """
    ev_go = ev_go_for_it(own_yardline, ydstogo)
    ev_fg = ev_field_goal(own_yardline)
    ev_pt = ev_punt(own_yardline)

    options = {
        "go_for_it": ev_go,
        "field_goal": ev_fg,
        "punt": ev_pt,
    }

    # Pick the highest EV (skipping NaN)
    valid = {k: v for k, v in options.items() if pd.notna(v)}
    if not valid:
        return {**options, "recommendation": None}
    best = max(valid, key=valid.get)

    return {**options, "recommendation": best}


# ---------------------------------------------------------------------------
# 8. Demo: walk through some example situations
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DECISION MODEL DEMO — example situations")
print("=" * 70)

examples = [
    (50, 1, "4th-and-1 at midfield"),
    (50, 5, "4th-and-5 at midfield"),
    (50, 10, "4th-and-10 at midfield"),
    (75, 2, "4th-and-2 at opp 25 (FG range)"),
    (95, 1, "4th-and-goal from the 5"),
    (98, 2, "4th-and-goal from the 2"),
    (15, 3, "4th-and-3 from own 15"),
    (30, 8, "4th-and-8 from own 30"),
    (35, 4, "4th-and-4 from own 35 (Romer's 'dead zone')"),
]

for yl, ytg, label in examples:
    result = fourth_down_decision(yl, ytg)
    print(f"\n{label}:")
    print(f"  Go for it:  EV = {result['go_for_it']:+.2f}")
    print(f"  Field goal: EV = {result['field_goal']:+.2f}" if pd.notna(result["field_goal"]) else "  Field goal: N/A (out of range)")
    print(f"  Punt:       EV = {result['punt']:+.2f}")
    print(f"  → Recommendation: {result['recommendation'].upper()}")


# ---------------------------------------------------------------------------
# 9. Build the critical-yards-to-go curve (Romer's Figure 5)
# ---------------------------------------------------------------------------
# At each field position, what's the LARGEST yards-to-go value where going
# for it is still preferred over kicking? This is Romer's most famous output.

print("\n" + "=" * 70)
print("Building the critical yards-to-go curve...")
print("=" * 70)

critical_yards = []
for yl in range(1, 100):
    max_ytg_to_go_for = 0  # will be 0 if we should never go for it at this yard line
    for ytg in range(1, 16):
        ev_go = ev_go_for_it(yl, ytg)
        ev_fg = ev_field_goal(yl)
        ev_pt = ev_punt(yl)

        # Best kicking option
        kicking_evs = [v for v in [ev_fg, ev_pt] if pd.notna(v)]
        if not kicking_evs:
            continue
        best_kick = max(kicking_evs)

        if pd.notna(ev_go) and ev_go > best_kick:
            max_ytg_to_go_for = ytg

    critical_yards.append({"own_yardline": yl, "critical_ydstogo": max_ytg_to_go_for})

critical_df = pd.DataFrame(critical_yards)


# ---------------------------------------------------------------------------
# 10. Plot Romer's Figure 5
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# The "go for it" region
ax.fill_between(
    critical_df["own_yardline"],
    0,
    critical_df["critical_ydstogo"],
    alpha=0.3,
    color="green",
    label="Go for it (model recommends)",
)

# The dividing line
ax.plot(
    critical_df["own_yardline"],
    critical_df["critical_ydstogo"],
    color="darkgreen",
    linewidth=2,
)

# Reference lines
ax.axhline(1, color="gray", linestyle=":", linewidth=0.8)
ax.axhline(5, color="gray", linestyle=":", linewidth=0.8)
ax.axhline(10, color="gray", linestyle=":", linewidth=0.8)

ax.set_xlabel("Team's own yard line (1 = own goal, 99 = opp goal)")
ax.set_ylabel("Yards to go")
ax.set_title(
    "Critical Yards-to-Go for Going For It on 4th Down\n"
    "(Replication of Romer 2003, Figure 5)"
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 15)
ax.legend(loc="upper left")

# Annotate key regions
ax.text(50, 13, "Above this curve: kick", fontsize=11, color="dimgray", style="italic")
ax.text(50, 1.5, "Below this curve: go for it", fontsize=11, color="darkgreen", style="italic")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/06_decision_curve.png", dpi=150)
plt.show()


# ---------------------------------------------------------------------------
# 11. Summary table — when should you go for it?
# ---------------------------------------------------------------------------
print("\n--- Critical yards-to-go at key field positions ---")
print(f"{'Field position':<35} {'Max ydstogo to go for it'}")
print("-" * 60)
for yl, label in [
    (10, "Own 10 (deep in own territory)"),
    (25, "Own 25"),
    (40, "Own 40"),
    (50, "Midfield"),
    (60, "Opp 40"),
    (65, "Opp 35 (Romer's dead zone)"),
    (75, "Opp 25 (FG range)"),
    (85, "Opp 15"),
    (95, "Opp 5"),
]:
    max_ytg = critical_df.loc[critical_df["own_yardline"] == yl, "critical_ydstogo"].values[0]
    print(f"{label:<35} {max_ytg}")


# ---------------------------------------------------------------------------
# 12. Save the decision data
# ---------------------------------------------------------------------------
critical_df.to_csv("output/critical_ydstogo.csv", index=False)
print(f"\nDecision curve saved to output/critical_ydstogo.csv")
print(f"Plot saved to figures/06_decision_curve.png")
