"""
evaluate.py
-----------
Evaluate the trained Random Forest model and generate performance
metrics, confusion matrix, and ROC curve plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from train_model import main as train_main, load_model
from utils import save_plot, get_logger

logger = get_logger(__name__)


# ── Metrics ─────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """
    Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 45)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 45)
    for name, value in metrics.items():
        print(f"  {name:<15s} : {value:.4f}")
    print("=" * 45 + "\n")


# ── Confusion Matrix Plot ───────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    save_plot(fig, "confusion_matrix.png")
    plt.close(fig)


# ── ROC Curve Plot ──────────────────────────────────────────────────

def plot_roc_curve(y_true, y_proba, auc_score: float) -> None:
    """Generate and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"Random Forest (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    save_plot(fig, "roc_curve.png")
    plt.close(fig)


# ── Full evaluation pipeline ───────────────────────────────────────

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Run the complete evaluation: metrics + plots.

    Returns
    -------
    dict
        Computed metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics)

    # Classification report (detailed)
    logger.info(
        "\n%s",
        classification_report(
            y_test, y_pred, target_names=["No Default", "Default"]
        ),
    )

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"])

    return metrics


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """Train (or load) the model and evaluate on the test set."""
    model, artefacts = train_main()
    metrics = evaluate_model(model, artefacts["X_test"], artefacts["y_test"])
    return metrics


if __name__ == "__main__":
    main()
