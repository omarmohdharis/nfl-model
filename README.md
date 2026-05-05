# NFL 4th Down Decision Model

Building on David Romer (2003), "It's Fourth Down and What Does the Bellman Equation Say?"

## Goal

Replicate Romer's value-of-field-position estimation on modern NFL data, then extend with a novel angle (TBD — likely team-specific recommendations or coach decision analysis).

## Project Structure

```
.
├── 01_explore_pbp.py       # First exploratory script — pulls data, crude V plot
├── figures/                 # Generated plots
├── data/                    # (gitignored) cached data
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the first exploratory script
python 01_explore_pbp.py
```

## Roadmap

- [x] Set up repo and dependencies
- [x] **Week 1:** Crude exploration of value-of-field-position (current script)
- [x] **Week 1:** Full recursive V_i estimation (Romer footnote 2 approach)
- [x] **Week 2:** Three component models — FG success, punt distribution, conversion probability
- [x] **Week 2:** Combine into 4txh down recommendation engine
- [x] **Week 3+:** Pick novel extension (team-specific / coach analysis / opponent-adjusted)
- [x] Build interactive dashboard for recommendations

## References

- Romer, D. (2003). It's Fourth Down and What Does the Bellman Equation Say? *NBER Working Paper 9024*.
- nflfastR / nfl_data_py for play-by-play data
