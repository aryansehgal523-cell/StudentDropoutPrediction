"""
Load raw CSV, perform cleaning, imputation, encoding, scaling, and save processed train/val/test splits.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

RND = 42
BASE = Path(__file__).resolve().parents[2]
# Prefer a provided dataset if present (semicolon-separated CSV). Fall back to simulated students.csv.
RAW_PREFERRED = BASE / "data" / "raw" / "data.csv"
RAW_FALLBACK = BASE / "data" / "raw" / "students.csv"
if RAW_PREFERRED.exists():
    RAW = RAW_PREFERRED
else:
    RAW = RAW_FALLBACK
PROC_DIR = BASE / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading raw data from {RAW}")
# This dataset may be semicolon-separated and contain spaces in column names.
try:
    df = pd.read_csv(RAW, sep=';')
except Exception:
    df = pd.read_csv(RAW)

# Clean column names: strip whitespace
df.columns = [c.strip() for c in df.columns]

# Dataset uses 'Target' with values like 'Dropout', 'Graduate', 'Enrolled'.
TARGET = "Target"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in {RAW}")

# Binary target: Dropout vs Not
y = (df[TARGET].astype(str).str.strip().str.lower() == 'dropout').astype(int)

# Select a concise set of important features from the provided dataset.
# These were chosen to be representative, compact, and predictive.
selected_features = [
    'Age at enrollment',
    'Gender',
    'Admission grade',
    'Tuition fees up to date',
    'Scholarship holder',
    'Debtor',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'GDP'
]

# Keep only selected features that exist in the dataset (safe fallback)
existing = [c for c in selected_features if c in df.columns]
if not existing:
    raise ValueError("None of the selected features exist in the provided dataset.")

X = df[existing].copy()

# Train/val/test split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, stratify=y, test_size=0.3, random_state=RND)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, stratify=y_tmp, test_size=0.5, random_state=RND)

# Identify column types
# Recompute types after selection
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Fit preprocessor on training data and transform all splits
print("Fitting preprocessing pipeline...")
preprocessor.fit(X_train)

X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# Save processed arrays and the transformer
joblib.dump(preprocessor, PROC_DIR / "preprocessor.joblib")
joblib.dump((X_train_proc, y_train), PROC_DIR / "train.pkl")
joblib.dump((X_val_proc, y_val), PROC_DIR / "val.pkl")
joblib.dump((X_test_proc, y_test), PROC_DIR / "test.pkl")

# Save the columns mapping for later use (feature names after one-hot)
# Build feature names
num_features = numeric_cols
cat_features = []
try:
    if categorical_cols:
        # OneHotEncoder may not expose names if there are no categorical columns or not fitted
        cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = list(cat_ohe.get_feature_names_out(categorical_cols))
except Exception:
    cat_features = []

feature_names = num_features + cat_features

(pd.DataFrame({"feature": feature_names})
   .to_csv(PROC_DIR / "feature_names.csv", index=False))

print(f"Saved processed data to {PROC_DIR}")
