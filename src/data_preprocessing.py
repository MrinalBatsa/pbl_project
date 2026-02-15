"""
data_preprocessing.py
---------------------
Load, clean, and split the Credit Card Default dataset.

Key steps:
    1. Load raw CSV (skipping the extra generic header row).
    2. Drop the ID column.
    3. Rename / clean column names.
    4. Handle missing values.
    5. Encode categorical features.
    6. Scale numerical features.
    7. Train-test split (80/20).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COL,
    get_logger,
)

logger = get_logger(__name__)


# ── 1. Load ─────────────────────────────────────────────────────────

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the raw dataset from CSV.

    The UCI file shipped with two header rows; we skip the first
    (generic X1 … Y) and use the second as column names.

    Parameters
    ----------
    filepath : str, optional
        Full path to the CSV file.  Defaults to ``data/default_credit.csv``.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = DATA_DIR / "default_credit.csv"

    df = pd.read_csv(filepath, header=1)  # skip the generic header row
    logger.info("Dataset loaded — shape %s", df.shape)
    return df


# ── 2. Clean ────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning:
    - Drop the ID column.
    - Rename the target column to a Python-friendly name.
    - Replace undocumented category codes with NaN then impute.
    """
    df = df.copy()

    # Drop ID
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)

    # Standardise target column name
    if TARGET_COL in df.columns:
        df.rename(columns={TARGET_COL: "default"}, inplace=True)

    # EDUCATION: 0, 5, 6 are undocumented → group into "Other" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 0 is undocumented → group into "Other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Handle any remaining missing values with median (numeric) / mode (object)                
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ("float64", "int64"):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info("Cleaning complete — shape %s", df.shape)
    return df


# ── 3. Feature engineering helpers ──────────────────────────────────

# Columns already numeric & ordinal; no one-hot encoding needed for
# tree-based models.  We still scale for consistency & potential
# downstream use with other algorithms.

NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """
    Apply StandardScaler to the continuous numeric columns.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame

    Returns
    -------
    X_train_scaled, X_test_scaled : pd.DataFrame
        DataFrames with scaled numeric columns.
    scaler : StandardScaler
        Fitted scaler (kept for inverse-transforms later).
    """
    scaler = StandardScaler()
    cols_to_scale = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    logger.info("Scaled %d numeric features", len(cols_to_scale))
    return X_train, X_test, scaler


# ── 4. Split ────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple:
    """
    Split into features / target and then train / test sets.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    target = "default"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split → train %d | test %d  (positive rate %.2f%%)",
        len(X_train),
        len(X_test),
        y.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ── 5. Full pipeline ───────────────────────────────────────────────

def preprocess_pipeline(filepath: str = None) -> dict:
    """
    Run the entire preprocessing pipeline and return artefacts.

    Returns
    -------
    dict with keys:
        df_clean, X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(filepath)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_features(X_train, X_test)

    return {
        "df_clean": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    artefacts = preprocess_pipeline()
    print("Preprocessing complete.")
    print(f"  Training samples : {len(artefacts['X_train'])}")
    print(f"  Test samples     : {len(artefacts['X_test'])}")
    print(f"  Features         : {artefacts['X_train'].shape[1]}")
