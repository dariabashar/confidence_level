# Confidence Level Detection from Posture (ML)

End-to-end coursework: multiclass prediction of a person’s confidence level (**Confident**, **Neutral**, **Low**) from posture, head, shoulder, and eye measurements.

## Repository layout

`confidence_level_ipynb".ipynb` - main Jupyter notebook: EDA, feature engineering, sklearn pipelines, model training and comparison, error analysis, SVM hyperparameter tuning, final model choice |

Consider renaming the notebook to something like `confidence_level.ipynb` to avoid the stray quote in the filename.

## Data

Uses the **Confidence Detection** dataset (confidence-related pose and body features).

- **Source:** Muhammad Khubaib Ahmad. (2025). *Confidence Detection Dataset* [Data set]. Kaggle. [https://doi.org/10.34740/KAGGLE/DSV/13488575](https://doi.org/10.34740/KAGGLE/DSV/13488575)
- **Size:** ~5,949 rows × 19 columns (numeric measurements plus `head_direction`, `arm_position`, `posture`, target `confidence_label`).

The notebook loads data from:

```text
supervised learning/datasets/confidence_features.csv
```

Create that folder, place `confidence_features.csv` there after downloading from Kaggle, **or** change the path in the `pd.read_csv(...)` cell to match your local file.

## Run locally

1. Python 3.9+ (3.10–3.12 recommended).
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

3. Start Jupyter and open the notebook:

```bash
jupyter notebook
```

## Run in Google Colab

The first notebook cell includes an “Open in Colab” badge (points to the `dariabashar/confidence_level` repo). In Colab, upload the CSV or use the Kaggle API to download the dataset.

## Notebook contents

1. Data loading and quality checks (missing values, duplicates, dtypes).
2. **EDA:** distributions, correlations, class balance.
3. **Feature engineering:** `posture_stability`, `lean_ratio`, `eye_ratio_adjusted`, `body_tension` plus a custom transformer inside `sklearn.Pipeline`.
4. **Preprocessing:** `OneHotEncoder` for categoricals, stratified 80/20 split, `class_weight='balanced'` where applicable.
5. **Models:** Logistic Regression, Decision Tree, kNN, Random Forest, SVM (RBF), XGBoost; overfitting checks, confusion matrices, feature importance.
6. **GridSearchCV** for SVM (example tuning of `C`, `gamma`).
7. **Outcome:** Random Forest as the accuracy–simplicity–interpretability tradeoff (~98.7% test accuracy in the report); limitations and possible improvements.

## Dependencies

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `jupyter` (or VS Code / Cursor with the Jupyter extension)

## License and attribution

This project uses an external Kaggle dataset; cite the dataset author and DOI above when publishing or referencing the data.
