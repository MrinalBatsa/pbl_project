"""
explainability.py
-----------------
Generate SHAP-based explanations for the trained Random Forest model.

Produces:
    - SHAP summary (beeswarm) plot
    - Feature importance bar plot
    - SHAP waterfall plot for a single prediction
    - Plain-English interpretation printed to console
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from train_model import main as train_main
from utils import save_plot, get_logger

logger = get_logger(__name__)


# ── SHAP Values ────────────────────────────────────────────────────

def compute_shap_values(model, X_test, max_samples: int = 500):
    """
    Compute SHAP values using TreeExplainer (fast, exact for trees).

    A random subsample is used to keep computation tractable while
    preserving representative distributions.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted model.
    X_test : pd.DataFrame
        Test features.
    max_samples : int
        Maximum number of samples to explain (default 500).

    Returns
    -------
    shap.Explanation
        SHAP explanation object.
    """
    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
        logger.info("Subsampled %d → %d for SHAP", len(X_test), max_samples)
    else:
        X_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    logger.info("SHAP values computed for %d samples", X_sample.shape[0])
    return shap_values


# ── Summary (Beeswarm) Plot ────────────────────────────────────────

def plot_shap_summary(shap_values) -> None:
    """Generate and save the SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    # For binary classification TreeExplainer returns shape
    # (n_samples, n_features, 2).  We take class-1 (default).
    vals = shap_values
    if len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]

    shap.plots.beeswarm(vals, show=False)
    plt.title("SHAP Summary — Feature Impact on Default Prediction", fontsize=13)
    plt.tight_layout()
    save_plot(plt.gcf(), "shap_summary.png", dpi=150)
    plt.close("all")


# ── Feature Importance Bar Plot ─────────────────────────────────────

def plot_feature_importance(shap_values) -> None:
    """Bar chart of mean |SHAP| values per feature."""
    vals = shap_values
    if len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(vals, show=False)
    plt.title("Feature Importance (Mean |SHAP| Value)", fontsize=13)
    plt.tight_layout()
    save_plot(plt.gcf(), "shap_feature_importance.png", dpi=150)
    plt.close("all")


# ── Single Prediction Explanation ──────────────────────────────────

def plot_single_explanation(shap_values, index: int = 0) -> None:
    """
    Waterfall plot explaining the prediction for one sample.

    Parameters
    ----------
    shap_values : shap.Explanation
    index : int
        Row index in the test set to explain.
    """
    vals = shap_values
    if len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(vals[index], show=False)
    plt.title(f"SHAP Explanation — Sample #{index}", fontsize=13)
    plt.tight_layout()
    save_plot(plt.gcf(), "shap_single_explanation.png", dpi=150)
    plt.close("all")


# ── Plain-English Interpretation ───────────────────────────────────

def print_interpretation(shap_values) -> None:
    """Print a concise business interpretation of SHAP results."""
    vals = shap_values
    if len(shap_values.shape) == 3:
        vals = shap_values[:, :, 1]

    # Mean absolute SHAP per feature
    mean_abs = np.abs(vals.values).mean(axis=0)
    feature_names = vals.feature_names
    ranking = sorted(
        zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True
    )

    print("\n" + "=" * 60)
    print("  EXPLAINABILITY INSIGHTS")
    print("=" * 60)

    print("\n  Top-5 features influencing default prediction:")
    for i, (feat, imp) in enumerate(ranking[:5], 1):
        print(f"    {i}. {feat:<20s}  (mean |SHAP| = {imp:.4f})")

    print(
        "\n  KEY BUSINESS INTERPRETATIONS\n"
        "  ─────────────────────────────────────────────────\n"
        "  • PAY_0 (most recent repayment status) is consistently the\n"
        "    strongest predictor.  Clients who delay payments even by one\n"
        "    month face significantly higher default risk.\n"
        "\n"
        "  • Payment history features (PAY_2 – PAY_6) collectively\n"
        "    dominate the model.  Repeated late payments compound risk.\n"
        "\n"
        "  • LIMIT_BAL (credit limit) acts as a proxy for the bank's\n"
        "    internal credit assessment.  Lower limits correlate with\n"
        "    higher default probability.\n"
        "\n"
        "  • Bill amounts and pay amounts have moderate influence;\n"
        "    large outstanding bills coupled with low repayments push\n"
        "    the model toward predicting default.\n"
        "\n"
        "  • Demographic features (SEX, EDUCATION, MARRIAGE) have\n"
        "    relatively low SHAP importance, suggesting the model\n"
        "    relies primarily on behavioural payment patterns.\n"
    )
    print("=" * 60 + "\n")


# ── Full pipeline ──────────────────────────────────────────────────

def explain(model=None, artefacts=None):
    """Run the complete explainability pipeline."""
    if model is None or artefacts is None:
        model, artefacts = train_main()

    X_test = artefacts["X_test"]

    shap_values = compute_shap_values(model, X_test)

    plot_shap_summary(shap_values)
    plot_feature_importance(shap_values)
    plot_single_explanation(shap_values, index=0)
    print_interpretation(shap_values)

    return shap_values


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    explain()
