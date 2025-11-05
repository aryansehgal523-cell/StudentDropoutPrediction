"""
Utility functions for training and evaluation: metrics, plotting helpers.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

sns.set(style="whitegrid")


def evaluate_and_persist(y_true, y_pred, y_proba, out_prefix):
    """Compute common metrics and save confusion matrix and ROC data.

    Args:
        y_true: array-like true labels (0/1)
        y_pred: predicted labels (0/1)
        y_proba: predicted probabilities for positive class
        out_prefix: Path or string prefix for saving plots (without extension)
    Returns:
        dict of metrics
    """
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics["roc_auc"] = np.nan

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{out_prefix}_confusion.png", bbox_inches="tight")
    plt.close()

    # Save ROC curve data (caller can plot if needed)
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        np.savez(f"{out_prefix}_roc.npz", fpr=fpr, tpr=tpr)
    except Exception:
        pass

    return metrics


def plot_feature_importance(importances, feature_names, out_path, top_n=20):
    if len(importances) != len(feature_names):
        # try handling when importances is dict or Series
        try:
            importances = np.array(importances)
        except Exception:
            return
    inds = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, min(6, 0.3*len(inds))))
    sns.barplot(x=importances[inds], y=np.array(feature_names)[inds], palette="viridis")
    plt.title("Feature importance (top {})".format(top_n))
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
