"""
Train multiple models and evaluate them. Saves best models and evaluation outputs to `outputs/`.
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data" / "processed"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

print("Loading processed data...")
preprocessor = joblib.load(PROC / "preprocessor.joblib")
X_train, y_train = joblib.load(PROC / "train.pkl")
X_val, y_val = joblib.load(PROC / "val.pkl")
X_test, y_test = joblib.load(PROC / "test.pkl")

models = {}
results = {}

# Train a compact set of models with class-balance handling to keep this realtime-friendly.
print("Training Logistic Regression (class_weight='balanced')...")
clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
params = {"C": [0.1, 1.0]}
grid = GridSearchCV(clf, params, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
models["logreg"] = grid.best_estimator_
print("best logreg:", grid.best_params_)

print("Training Random Forest (class_weight='balanced')...")
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
params = {"n_estimators": [100], "max_depth": [5, None]}
grid = GridSearchCV(clf, params, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
models["rf"] = grid.best_estimator_
print("best rf:", grid.best_params_)

# Evaluate on validation set and pick best by F1
best_name = None
best_f1 = -1
for name, m in models.items():
    print(f"Evaluating {name}...")
    y_pred = m.predict(X_val)
    try:
        y_proba = m.predict_proba(X_val)[:,1]
    except Exception:
        y_proba = np.zeros_like(y_pred, dtype=float)
    report = classification_report(y_val, y_pred, output_dict=True)
    f1 = report.get("1", {}).get("f1-score") or report.get("macro avg", {}).get("f1-score", 0)
    results[name] = {"f1": f1, "report": report}
    if f1 is not None and f1 > best_f1:
        best_f1 = f1
        best_name = name

print("Best model on validation:", best_name, best_f1)

# Retrain best model on train+val
print("Retraining best model on train+val data...")
X_train_val = np.vstack([X_train, X_val])
y_train_val = np.concatenate([y_train, y_val])
best_model = models[best_name]
best_model.fit(X_train_val, y_train_val)

# Final evaluation on test
print("Evaluating final model on test set...")
y_test_pred = best_model.predict(X_test)
try:
    y_test_proba = best_model.predict_proba(X_test)[:,1]
except Exception:
    y_test_proba = np.zeros_like(y_test_pred, dtype=float)

final_report = classification_report(y_test, y_test_pred, output_dict=True)
print(pd.DataFrame(final_report).transpose())

# Save model and results
joblib.dump(best_model, OUT / "best_model.joblib")
joblib.dump(results, OUT / "model_results.json")

# If RandomForest was used, save a simple feature importance plot (requires feature names)
try:
    if best_name == 'rf':
        import numpy as _np
        from src.models.utils import plot_feature_importance
        feat_df = pd.read_csv(PROC / "feature_names.csv")
        feature_names = feat_df['feature'].tolist()
        importances = best_model.feature_importances_
        plot_feature_importance(importances, feature_names, OUT / 'feature_importance.png')
        print('Saved feature importance to outputs/feature_importance.png')
except Exception as e:
    print('Could not save feature importance:', e)

print(f"Saved best model to {OUT / 'best_model.joblib'} and results to {OUT}")
