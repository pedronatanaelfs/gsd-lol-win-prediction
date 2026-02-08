# Gameplay Outcome Modeling — Early Game to Win Probability (LoL)

Predict match outcome using early-game signals and explain what drives wins. This project is structured like a game analytics deliverable: reproducible pipeline, clear evaluation, calibrated probabilities, and decision-oriented insights.

## Project goals
- Train a model that predicts **Blue team win / loss** using **early-match features**.
- Produce **well-calibrated win probabilities** (not just accuracy).
- Explain the prediction drivers with **interpretable global + local explanations** (e.g., SHAP).
- Translate results into actionable insights (what to prioritize early).

## Dataset
This project uses the Kaggle dataset:

- **League of Legends Diamond Ranked Games (10 min)**  
  https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min

The dataset includes match-level features captured around the first ~10 minutes (e.g., gold, objectives, kills, etc.) and a binary target indicating whether the Blue team won.

## What you’ll find in this repo
```
.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_calibration_and_thresholds.ipynb
│   └── 04_explanations_shap.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── eval.py
│   └── viz.py
├── reports/
│   ├── model_card.md
│   └── insights_report.pdf
├── assets/
│   └── figures/
├── requirements.txt
└── README.md
```

## Method overview
### 1) Data preparation
- Load raw Kaggle file(s)
- Validate schema (expected columns, target integrity)
- Train/test split with reproducible random seed  
  - Default: stratified split  
  - Optional: time-based split if timestamps/patch proxies are present

### 2) Feature engineering
Features are used as provided by the dataset (early-match signals), with optional derived features such as:
- Objective advantage deltas (Blue - Red)
- Resource advantage ratios (e.g., gold share)
- Normalized rates (per minute proxies if time is available)

All transformations are fit on training data only.

### 3) Modeling
Baseline to production-like:
- Baseline: Logistic Regression (interpretable benchmark)
- Strong baseline: Gradient Boosting (e.g., XGBoost/LightGBM or sklearn HistGB)
- Compare models on discrimination + calibration

### 4) Probability calibration
Since this is a “win probability” task, calibration is a first-class requirement:
- Reliability curve (calibration plot)
- Brier score
- Calibration methods: Platt scaling (sigmoid) and isotonic regression
- Select a calibrated model for reporting

### 5) Interpretability (SHAP)
Explain the model at two levels:
- Global: top drivers of win probability across matches
- Local: why the model predicted a specific match outcome (example slices)

## Evaluation
Primary metrics:
- ROC-AUC
- PR-AUC (useful if imbalance exists)
- Log loss
- Brier score (probability quality)
- Calibration error (ECE or similar)

Secondary checks:
- Confusion matrix at selected operating points
- Threshold analysis based on use-case (e.g., “call the game” confidence)

## Deliverables
### Model card
A concise summary in `reports/model_card.md`:
- Data + target definition
- Training setup and leakage checks
- Metrics (with calibration)
- Limitations and recommended use

### Evaluation notebook
Notebook(s) showing:
- EDA and sanity checks
- Model comparison
- Calibration choice and plots
- SHAP explanations and slices

### Insights report (PDF)
A short report in `reports/insights_report.pdf` with:
- What early signals most influence win probability
- How win probability changes with common early advantages
- Recommendations framed as gameplay priorities (with caveats)

## Quickstart
### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Download data
Option A (recommended): Use Kaggle CLI:
```bash
kaggle datasets download -d bobbyscience/league-of-legends-diamond-ranked-games-10-min -p data/raw --unzip
```

Option B: Download manually from Kaggle and place the raw CSV in:
```
data/raw/
```

### 3) Run notebooks
Start with:
- `notebooks/01_eda.ipynb`
- `notebooks/02_modeling.ipynb`
- `notebooks/03_calibration_and_thresholds.ipynb`
- `notebooks/04_explanations_shap.ipynb`

## Notes on leakage and realism
This project is careful about:
- Avoiding target leakage (no features computed using post-10-minute events)
- Ensuring derived features are built from early-game fields only
- Keeping evaluation honest (calibration on held-out data)

## Results (to be filled after training)
This section is intentionally left blank until the model is trained and the final calibrated model is selected.

- Best model:
- ROC-AUC:
- Brier score:
- Calibration method:
- Top drivers (global):

## License and attribution
- Dataset is provided by Kaggle and is subject to Kaggle’s dataset terms.
- This repository contains code and analysis only; it does not redistribute the raw dataset.
