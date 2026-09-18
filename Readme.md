# Cellphone Price Range Classification

A machine learning project that predicts a mobile phone's price category (low, medium, high, or very high) from its hardware and connectivity specifications.

## Overview

The dataset contains **2,000 phones** described by 20 features — battery power, clock speed, camera specs, memory, screen dimensions, processor cores, talk time, and connectivity flags (Bluetooth, dual-SIM, 3G/4G, Wi-Fi, touch screen) — labeled with a target `price_range`:

| Value | Price Range     |
|:-----:|------------------|
| 0     | Low cost         |
| 1     | Medium cost      |
| 2     | High cost        |
| 3     | Very high cost   |

The classes are perfectly balanced (500 phones each).

## Project Structure

```
.
├── data/
│   └── data.csv                     # source dataset (2000 rows x 21 columns)
├── images/
│   ├── eda plots/                   # EDA figures saved by the notebook
│   └── model_performance_plot/      # model comparison chart
├── model/
│   ├── model_result.csv             # metrics for all 9 candidate models
│   ├── confusion_matrix.csv         # confusion matrix of the best baseline model
│   └── model.pkl                    # final tuned model (pickled)
├── notebook/
│   └── Cellphone_Price_range.ipynb  # full analysis notebook
├── report.pdf                       # formatted write-up of the analysis and results
└── README.md
```

> The notebook expects to be run from a `notebook/` (or similar) working directory, since it reads `../data/data.csv` and writes figures to `../images/...` and model artifacts to `../model/...`. Create those folders (or adjust the paths) before running.

## Workflow

1. **Import libraries** — pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost.
2. **Load data** — read `data.csv`, inspect shape/dtypes.
3. **EDA**
   - Missing value / duplicate checks (both 0)
   - Target distribution (balanced across 4 classes)
   - Univariate analysis of numerical and categorical features
   - Outlier scan via IQR (negligible outliers, only in `fc` and `px_height`)
   - Correlation analysis — `ram` is by far the strongest predictor (r = 0.92)
4. **Train/test split** — 80/20 stratified split, features standardized with `StandardScaler`.
5. **Model selection** — 9 classifiers trained and compared via 5-fold stratified cross-validation and held-out test accuracy:
   Logistic Regression, KNN, SVM (linear & RBF), Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, XGBoost.
6. **Hyperparameter tuning** — `GridSearchCV` over the best baseline model (Logistic Regression), tuning the regularization strength `C`.
7. **Persist model** — final tuned model saved to `model/model.pkl` with `pickle`.

## Key Results

| Metric                          | Value                    |
|----------------------------------|---------------------------|
| Best baseline model               | Logistic Regression      |
| Baseline test accuracy            | 97.5%                    |
| Best hyperparameter               | C = 100                  |
| Best CV accuracy (tuned)          | 96.9%                    |
| Final tuned test accuracy         | 97.5%                    |
| Macro F1-score (tuned)            | 0.97                     |

RAM dominates as a predictor of price range; battery power and screen resolution (pixel width/height) contribute modestly, while connectivity flags (Bluetooth, Wi-Fi, dual-SIM, 3G/4G) carry little predictive signal. Most misclassifications occur between adjacent price classes, consistent with `price_range` being an ordinal variable.

See **`report.pdf`** for the full write-up with figures and tables.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Running the Notebook

1. Place `data.csv` in a `data/` folder one level above the notebook.
2. Ensure `images/eda plots/` and `images/model_performance_plot/` folders exist one level above the notebook (or update the `plt.savefig` paths).
3. Ensure a `model/` folder exists one level above the notebook for the saved CSVs and pickled model.
4. Run all cells top to bottom.