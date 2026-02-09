# Model Card

## Model Overview

**Model Name:** Logistic Regression with Platt Scaling Calibration  
**Task:** Binary classification - Predicting Blue team win probability in League of Legends matches  
**Model Type:** Calibrated Logistic Regression  
**Date:** 2024

## Data + Target Definition

### Dataset
- **Source:** Kaggle - League of Legends Diamond Ranked Games (10 min)
- **Dataset Size:** 9,879 matches
- **Time Period:** Diamond rank games captured at ~10 minutes
- **Data Quality:** No missing values, no duplicate gameIds

### Target Variable
- **Name:** `blueWins`
- **Type:** Binary (0 = Blue team loses, 1 = Blue team wins)
- **Distribution:** Approximately balanced (50.1% Blue wins, 49.9% Blue loses)
- **Definition:** Whether the Blue team won the match (determined post-game)

### Features
- **Total Features:** 50 (38 original + 12 engineered)
- **Feature Categories:**
  - **Combat:** Kills, deaths, assists, first blood
  - **Objectives:** Dragons, heralds, elite monsters, towers destroyed
  - **Resources:** Total gold, gold per minute, total experience, experience differences
  - **Vision:** Wards placed, wards destroyed
  - **Farming:** Minions killed, jungle minions killed, CS per minute
  - **Level:** Average champion level

### Feature Engineering
- Created advantage features (Blue - Red deltas) for:
  - Gold, experience, kills, deaths, assists
  - Objectives (dragons, heralds, towers, elite monsters)
  - Vision (wards placed/destroyed)
  - Farming (CS, jungle CS)
  - Level differences

### Data Split
- **Training Set:** 7,903 samples (80%)
- **Test Set:** 1,976 samples (20%)
- **Split Method:** Stratified random split (preserves class distribution)
- **Random Seed:** 42 (for reproducibility)

## Training Setup and Leakage Checks

### Preprocessing
- **Scaling:** StandardScaler applied to all features (required for Logistic Regression)
- **Feature Selection:** All features used (no feature selection performed)
- **Handling:** No missing values or outliers removed

### Model Training
- **Base Model:** Logistic Regression
  - Solver: LBFGS
  - Max iterations: 1000
  - Random state: 42
- **Calibration:** Platt Scaling (sigmoid calibration)
  - Method: CalibratedClassifierCV with 5-fold cross-validation
  - Calibration fit on training data only

### Leakage Prevention
- **Temporal Leakage:** All features captured at ~10 minutes (early game only)
- **Target Leakage:** No features computed using post-10-minute events
- **Data Leakage:** Train/test split performed before any feature engineering
- **Calibration Leakage:** Calibration fit on training data, evaluated on held-out test set
- **Cross-Validation:** Used for calibration only, not for model selection

### Model Selection
- Compared two models:
  1. Logistic Regression (baseline)
  2. HistGradientBoostingClassifier (strong baseline)
- Selected based on combined score: ROC-AUC - (ECE + Brier Score)/2
- Best model: Logistic Regression with Platt calibration

## Metrics (with Calibration)

### Discrimination Metrics (Test Set)
- **ROC-AUC:** 0.8058
- **PR-AUC:** 0.8049
- **Log Loss:** 0.5316
- **Brier Score:** 0.1798

### Calibration Metrics (Test Set)
- **Expected Calibration Error (ECE):** 0.0225
- **Brier Score:** 0.1798
- **Calibration Method:** Platt Scaling (sigmoid)

### Performance Summary
- **Discrimination:** Good discrimination (ROC-AUC > 0.80)
- **Calibration:** Well-calibrated (ECE < 0.03)
- **Probability Quality:** Reliable win probabilities suitable for decision-making

### Comparison with Alternatives
| Model | Calibration | ROC-AUC | ECE | Brier Score |
|-------|-------------|---------|-----|-------------|
| Logistic Regression | Uncalibrated | 0.8058 | 0.0260 | 0.1799 |
| Logistic Regression | **Platt** | **0.8058** | **0.0225** | **0.1798** |
| Logistic Regression | Isotonic | 0.8053 | 0.0266 | 0.1802 |
| HistGradientBoosting | Uncalibrated | 0.8024 | 0.0272 | 0.1819 |
| HistGradientBoosting | Platt | 0.8048 | 0.0264 | 0.1805 |
| HistGradientBoosting | Isotonic | 0.8049 | 0.0300 | 0.1805 |

### Threshold Analysis
- **Default Threshold (0.5):** Balanced precision and recall
- **Best F1-Score Threshold:** 0.30 (F1 = 0.7449)
- **High Confidence (0.7+):** High precision scenarios
- **Low Confidence (0.3-):** High recall scenarios

## Model Interpretability

### Top Features (SHAP Analysis)
1. Gold differences (blueGoldDiff, redGoldDiff)
2. Total gold (blueTotalGold, redTotalGold)
3. Gold per minute (blueGoldPerMin, redGoldPerMin)
4. Total experience (redTotalExperience)
5. Experience differences (blueExperienceDiff, redExperienceDiff)
6. Objectives (blueDragons, blueDragonAdvantage)

### Key Insights
- **Economic factors** (gold, experience) are the strongest predictors
- **Advantage features** (Blue - Red deltas) are highly informative
- **Early game resource accumulation** is critical for win prediction
- Model uses a combination of combat, economic, and objective metrics

## Limitations and Recommended Use

### Limitations

1. **Temporal Scope:**
   - Model only uses data from first ~10 minutes
   - Cannot account for mid/late game events or comebacks
   - Predictions are static snapshots, not dynamic updates

2. **Rank Restriction:**
   - Trained on Diamond rank games only
   - May not generalize to other ranks (Bronze, Silver, Gold, Platinum, Master+)
   - Meta and playstyle differences across ranks not captured

3. **Feature Limitations:**
   - No champion-specific information
   - No player skill/performance history
   - No team composition synergies
   - No patch/meta information

4. **Model Limitations:**
   - Linear model may miss complex interactions
   - Assumes feature relationships are consistent across matches
   - Cannot capture non-linear patterns that tree-based models might find

5. **Data Limitations:**
   - Single dataset source (Kaggle)
   - Potential selection bias in data collection
   - No information about match context (tournament, ranked, normal)

6. **Calibration:**
   - Calibration performed on held-out test set
   - May degrade over time as game meta evolves
   - Requires periodic recalibration

### Recommended Use Cases

1. **Early Game Analysis:**
   - Assess win probability at 10-minute mark
   - Identify critical early game factors
   - Educational/training purposes for players

2. **Broadcast/Analytics:**
   - Provide win probability estimates during live games
   - Generate insights for commentary
   - Post-game analysis and statistics

3. **Research:**
   - Study early game impact on outcomes
   - Feature importance analysis
   - Understanding game mechanics

### Not Recommended For

1. **Betting/Gambling:**
   - Model not designed for financial decisions
   - Probabilities are estimates, not guarantees
   - Ethical concerns with gambling applications

2. **Real-time In-Game Decisions:**
   - Model uses 10-minute snapshot, not real-time updates
   - Not suitable for dynamic decision-making during matches

3. **Other Ranks:**
   - Model trained on Diamond rank only
   - Performance may degrade significantly for other ranks

4. **Long-term Predictions:**
   - Model designed for 10-minute predictions
   - Cannot predict final outcome with high confidence beyond early game

### Maintenance and Updates

1. **Periodic Recalibration:**
   - Recalibrate as game meta evolves
   - Monitor calibration metrics on new data
   - Update if ECE exceeds 0.05

2. **Performance Monitoring:**
   - Track ROC-AUC on new data
   - Monitor for distribution shift
   - Retrain if performance degrades significantly

3. **Feature Updates:**
   - Update features if game mechanics change
   - Add new features if available (e.g., new objectives)
   - Remove deprecated features

### Ethical Considerations

- **Fair Use:** Model should be used for educational and analytical purposes
- **No Harm:** Should not be used to exploit players or manipulate outcomes
- **Transparency:** Model limitations should be clearly communicated
- **Privacy:** No personal player data used in model

## Model Files and Reproducibility

### Reproducibility
- **Random Seed:** 42 (used consistently across all notebooks)
- **Data Split:** Stratified 80/20 split with random_state=42
- **Code:** All notebooks available in `notebooks/` directory
- **Dependencies:** See `requirements.txt`

### Model Artifacts
- Model can be recreated by running notebooks in sequence:
  1. `01_eda.ipynb` - Data exploration
  2. `02_modeling.ipynb` - Model training
  3. `03_calibration_and_thresholds.ipynb` - Calibration
  4. `04_explanations_shap.ipynb` - Interpretability

### Version Information
- **Python:** 3.9+
- **scikit-learn:** 1.2.0+
- **SHAP:** 0.41.0+
- **Other dependencies:** See `requirements.txt`

## Contact and Attribution

- **Dataset:** Kaggle - League of Legends Diamond Ranked Games (10 min)
- **Dataset Creator:** bobbyscience
- **Dataset License:** Subject to Kaggle's dataset terms
- **Model Implementation:** This repository (code only, no data redistribution)

---
