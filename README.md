# Fraud Detection ML

A complete end-to-end pipeline for credit-card fraud detection using tree-based models. From raw data exploration through production-ready model serving, this repo contains:

- **Notebooks** for EDA, modeling, and explainability  
- **Scripts** for preprocessing, training, and evaluation  
- **Streamlit app** for interactive demo  
- **Reports** with saved charts and metrics  

---

## Quick Start

### 1. Clone & install dependencies  
```bash
git clone <repo-url>
cd fraud-detection-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the data
Place the original `creditcard.csv` in `Data/`.
Then run:

```bash
python src/preprocessing.py \
  --input_csv Data/creditcard.csv \
  --out_prefix Data/prep/creditcard
```

This will:
- Drop duplicates
- Auto‐log-transform skewed Tier-1 features
- Subset to your selected V-features + target
- Split into train/val/test .npy files under `Data/prep/`

### 3. Train your champion model

```bash
python src/train.py \
  --data_prefix Data/prep/creditcard \
  --model_path Models/rf_champion.joblib
```

This will re‐load train+val, fit your RandomForest (or LightGBM), and save the final model.

### 4. Evaluate on hold-out test set

```bash
python src/evaluate.py \
  --model_path Models/rf_champion.joblib \
  --data_prefix Data/prep/creditcard \
  --out_dir reports/evaluation
```

Generates:
- ROC curve & AUC
- Precision–Recall curve & average precision
- Confusion matrix (legitimate vs fraudulent)
- Precision@K (1%, 5%, 10%)
- Saved charts & a `metrics_summary.txt` in `reports/evaluation/`

### 5. Explainability
Open and run `notebooks/03_explainability.ipynb` (or `src/explainability.py`) to produce:
- SHAP global bar & dot plots
- Partial Dependence Plots for top features
- Local force plots for high-risk samples
- Narrative bullets summarizing "why the model does what it does"

Artifacts are saved under `reports/explainability/`.

### 6. Interactive demo

```bash
streamlit run app/streamlit_app.py
```

Fill in transaction fields in your browser and see live fraud predictions + SHAP explanations.

---

## 📂 Repository Structure

```
├── app/
│   └── streamlit_app.py         # Streamlit UI for live serving
├── Data/
│   ├── creditcard.csv           # Raw Kaggle dataset
│   └── prep/                    # Preprocessed train/val/test .npy
├── Models/
│   └── rf_champion.joblib       # Final trained model artifact
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_modeling.ipynb        # Baseline & hyperparameter tuning
│   └── 03_explainability.ipynb  # SHAP & PDP explainability
├── reports/
│   ├── evaluation/              # Saved metrics & plots
│   └── explainability/          # SHAP & PDP figures
├── src/
│   ├── preprocessing.py         # Data cleaning & split
│   ├── train.py                 # Train on train+val & save model
│   └── evaluate.py              # Load model, test eval & charts
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore                   
```

---

## Key Components

### Tier-1 features
Only the most statistically significant V-columns (V2, V3, V4, V7, V10, V11, V12, V14, V16, V17, V18) are used to minimize noise.

### Preprocessing
- Drops duplicates
- Auto log-transforms skewed, non-negative features
- Splits data time-aware into train/val/test

### Modeling
- Baselines: Dummy, DecisionTree, RandomForest, LightGBM
- Hyperparameter tuning via RandomizedSearchCV or Optuna
- Champion selection by validation ROC-AUC

### Evaluation
- ROC & PR curves
- Confusion matrix with "Legitimate"/"Fraudulent" labels
- Precision@K metrics

### Explainability
- Global SHAP importance (bar & dot)
- Partial Dependence Plots (PDP) for the top 3 features
- Local SHAP force-plots for the highest-risk cases

---

## Packaging & Deployment

- Containerize with Docker for reproducible environments
- Deploy the Streamlit app to a cloud VM (Heroku, AWS EC2)
- Automate daily data ingestion & scoring via Airflow or cron

---

## License

This project is released under the MIT License.
Feel free to use, modify, or extend!