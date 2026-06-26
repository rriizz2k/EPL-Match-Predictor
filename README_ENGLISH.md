# EPL Match Outcome Predictor

A machine learning model that predicts English Premier League match outcomes (Home Win / Draw / Away Win) based on historical team statistics.

## Results

| Metric | Value |
|---|---|
| Accuracy | **53.5%** |
| Baseline (naive "always home win" strategy) | 46.5% |
| Algorithm | Random Forest Classifier |
| Train/Test split | 70/30, chronological |
| Season | 2023-24 (Premier League) |

The model outperforms the naive baseline by ~7 percentage points, confirming a real predictive signal in the engineered features.

### Confusion Matrix

| | Predicted: H | Predicted: D | Predicted: A |
|---|---|---|---|
| **Actual: H** | 34 | 5 | 7 |
| **Actual: D** | 13 | 2 | 9 |
| **Actual: A** | 10 | 5 | 14 |

**Known limitation:** the model struggles significantly with draw (D) prediction — recall for this class is very low (2 out of 24). This is a well-documented challenge in sports outcome prediction: draws are statistically "blurred" between two decisive outcomes and lack a clear signal in team-form data.

## Features

All features are computed using a rolling 5-match window per team, with no data leakage from future matches (`.shift(1)` applied after `.rolling()`):

- `Form_Home`, `Form_Away` — points earned in the last 5 matches
- `Scored_5g_Home`, `Scored_5g_Away` — goals scored in the last 5 matches
- `Conceded_5g_Home`, `Conceded_5g_Away` — goals conceded in the last 5 matches
- `Form_Diff`, `Scored_Diff`, `Conceded_Diff` — differences in the metrics above between home and away teams

## Data Source

[football-data.co.uk](https://www.football-data.co.uk) — historical Premier League match results, 2023-24 season.

## Tested and Rejected Hypothesis: Multi-Season Data

**Hypothesis:** adding the 2022-23 season would expand the training set and improve draw prediction (the most underrepresented class).

**Implementation:** combined two seasons using `sample_weight`, assigning a lower weight (0.5) to the older season relative to the current one (1.0), to partially account for squad changes (transfers, managerial changes) between seasons.

**Result:** the hypothesis was rejected. Accuracy dropped from 53.5% (single season) to 46-48% (two seasons, both with and without weighting). The likely cause is that squad differences between seasons introduce more noise than the benefit gained from a larger sample size.

**Conclusion:** for this task, data quality and consistency mattered more than raw volume. The experiment is preserved in a separate notebook (`epl_two_seasons_experiment.ipynb`) for process transparency.

## How to Run

### Requirements

```
pandas
scikit-learn
```

Install:

```bash
pip install pandas scikit-learn
```

### Usage

Open `epl_prediction.ipynb` in Jupyter Notebook or Google Colab and run the cells sequentially. Data is loaded automatically via URL in the first cell — no manual file download required.

## Repository Structure

```
.
├── epl_prediction.ipynb              # main project (single season, final version)
├── epl_two_seasons_experiment.ipynb  # multi-season hypothesis experiment
├── README.md                         # Russian version
└── README_ENGLISH.md                 # this file
```

## Future Work

- Adding a "league table position at match time" feature
- Weighted head-to-head statistics between specific teams
- Hyperparameter tuning (`n_estimators`, `max_depth`)
- Addressing the draw-prediction weakness (alternative metrics or resampling)

## On the Development Process

All code was written independently by the author. AI tools were used strictly for educational purposes — explaining concepts, identifying logical errors, and guiding the learning process ("hints, not solutions"). No line of code was AI-generated directly.

## License

MIT License — see [LICENSE](LICENSE).
