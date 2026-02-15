"""
train_model.py
--------------
Train a Random Forest classifier on the preprocessed credit-default
dataset and persist the trained model to disk.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import preprocess_pipeline
from utils import MODELS_DIR, RANDOM_STATE, ensure_dirs, get_logger

logger = get_logger(__name__)


# ── Hyperparameters ─────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ── Training ────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, params: dict = None):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    params : dict, optional
        Override default hyperparameters.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    if params is None:
        params = RF_PARAMS

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained — %d trees, max_depth=%s",
                params["n_estimators"], params.get("max_depth"))
    return model


def save_model(model, filename: str = "random_forest.joblib") -> str:
    """Persist the trained model to ``outputs/models/``."""
    ensure_dirs()
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)
    return str(path)


def load_model(filename: str = "random_forest.joblib"):
    """Load a previously saved model."""
    path = MODELS_DIR / filename
    model = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return model


# ── CLI entry point ────────────────────────────────────────────────

def main():
    """Run the full training pipeline."""
    artefacts = preprocess_pipeline()

    model = train_random_forest(
        artefacts["X_train"],
        artefacts["y_train"],
    )

    save_model(model)
    return model, artefacts


if __name__ == "__main__":
    main()
