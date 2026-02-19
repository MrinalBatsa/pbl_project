"""
app.py
------
Interactive Streamlit dashboard for live demonstration of the
Credit Default Prediction + Explainability pipeline.

Run with:
    streamlit run app.py
"""

import sys
import os

# ── Ensure src/ is importable ───────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)

from src.data_preprocessing import preprocess_pipeline
from src.utils import MODELS_DIR, PLOTS_DIR, ensure_dirs

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default — XAI Dashboard",
    page_icon="🏦",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #60a5fa; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px 20px;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache heavy work ────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model & data …")
def load_everything():
    """Load data, train (or load) the model, and compute SHAP values."""
    ensure_dirs()
    artefacts = preprocess_pipeline()

    model_path = MODELS_DIR / "random_forest.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        from src.train_model import train_random_forest, save_model
        model = train_random_forest(artefacts["X_train"], artefacts["y_train"])
        save_model(model)

    X_test = artefacts["X_test"]
    y_test = artefacts["y_test"]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # SHAP on a subsample
    X_shap = X_test.sample(n=min(500, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)

    return {
        "model": model,
        "artefacts": artefacts,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "explainer": explainer,
        "shap_values": shap_values,
        "X_shap": X_shap,
    }


data = load_everything()
model = data["model"]
artefacts = data["artefacts"]
X_test = artefacts["X_test"]
y_test = artefacts["y_test"]
y_pred = data["y_pred"]
y_proba = data["y_proba"]
shap_vals = data["shap_values"]
X_shap = data["X_shap"]

# Handle 3-D SHAP (binary classification → take class-1 slice)
sv = shap_vals
if len(shap_vals.shape) == 3:
    sv = shap_vals[:, :, 1]

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown("## 🏦  Credit Card Default — Explainable AI Dashboard")
st.caption("Random Forest · SHAP · Live Interactive Demo")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🔍 SHAP Explainability",
    "🧪 Predict a Client",
    "📁 Dataset Explorer",
])

# ── TAB 1 — PERFORMANCE ────────────────────────────────────────────
with tab1:
    st.subheader("Evaluation Metrics")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val in zip(
        [c1, c2, c3, c4, c5],
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        [acc, prec, rec, f1, auc],
    ):
        col.metric(label, f"{val:.4f}")

    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Default", "Default"],
                    yticklabels=["No Default", "Default"], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_roc:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, color="#3b82f6", lw=2,
                    label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)
        plt.close(fig_roc)

# ── TAB 2 — SHAP ───────────────────────────────────────────────────
with tab2:
    st.subheader("SHAP Feature Explanations")

    shap_tab1, shap_tab2, shap_tab3 = st.tabs([
        "Summary (Beeswarm)", "Feature Importance", "Single Prediction"
    ])

    with shap_tab1:
        st.markdown("Each dot is one client. **Red** = high feature value, "
                     "**Blue** = low. Position shows push toward (right) "
                     "or away from (left) default.")
        fig_bee, ax_bee = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(sv, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

    with shap_tab2:
        st.markdown("Average absolute SHAP value per feature — "
                     "the higher the bar, the more important.")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 7))
        shap.plots.bar(sv, show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

    with shap_tab3:
        idx = st.slider("Select sample index", 0, len(X_shap) - 1, 0)
        st.markdown(f"**Explaining sample #{idx}** — each bar shows how "
                     "a feature pushed the prediction up (red) or down (blue).")
        fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(sv[idx], show=False)
        st.pyplot(plt.gcf())
        plt.close("all")

# ── TAB 3 — LIVE PREDICT ───────────────────────────────────────────
with tab3:
    st.subheader("🧪  Predict Default for a New Client")
    st.caption("Adjust the sliders to simulate a client profile and see "
               "the prediction with SHAP explanation in real time.")

    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            limit_bal = st.number_input("Credit Limit (NT$)", 10000, 1000000, 200000, step=10000)
            age = st.slider("Age", 20, 80, 35)
            sex = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = st.selectbox("Education", [1, 2, 3, 4],
                                     format_func=lambda x: {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}[x])
            marriage = st.selectbox("Marriage", [1, 2, 3],
                                    format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x])
        with col_b:
            st.markdown("**Repayment Status** (-1 = paid duly, 1 = 1-mo delay, 2 = 2-mo …)")
            pay_0 = st.slider("PAY_0 (Sept)", -2, 8, 0)
            pay_2 = st.slider("PAY_2 (Aug)", -2, 8, 0)
            pay_3 = st.slider("PAY_3 (Jul)", -2, 8, 0)
            pay_4 = st.slider("PAY_4 (Jun)", -2, 8, 0)
            pay_5 = st.slider("PAY_5 (May)", -2, 8, 0)
            pay_6 = st.slider("PAY_6 (Apr)", -2, 8, 0)
        with col_c:
            bill_1 = st.number_input("BILL_AMT1", 0, 500000, 50000, step=5000)
            bill_2 = st.number_input("BILL_AMT2", 0, 500000, 48000, step=5000)
            bill_3 = st.number_input("BILL_AMT3", 0, 500000, 45000, step=5000)
            pay_amt1 = st.number_input("PAY_AMT1", 0, 200000, 2000, step=1000)
            pay_amt2 = st.number_input("PAY_AMT2", 0, 200000, 2000, step=1000)
            pay_amt3 = st.number_input("PAY_AMT3", 0, 200000, 2000, step=1000)

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Build a single-row DataFrame matching training columns
        input_data = {
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
            "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
            "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill_1, "BILL_AMT2": bill_2, "BILL_AMT3": bill_3,
            "BILL_AMT4": 40000, "BILL_AMT5": 38000, "BILL_AMT6": 36000,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
            "PAY_AMT4": 1500, "PAY_AMT5": 1500, "PAY_AMT6": 1500,
        }
        input_df = pd.DataFrame([input_data])[X_test.columns]  # match column order

        # Scale using the same scaler
        from src.data_preprocessing import NUMERIC_FEATURES
        scaler = artefacts["scaler"]
        cols_to_scale = [c for c in NUMERIC_FEATURES if c in input_df.columns]
        input_scaled = input_df.copy()
        input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        prob = model.predict_proba(input_scaled)[0, 1]
        pred = "⚠️ DEFAULT" if prob >= 0.5 else "✅ NO DEFAULT"

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"### Prediction: {pred}")
            st.metric("Default Probability", f"{prob:.1%}")
        with res_col2:
            st.markdown("#### SHAP Explanation for This Client")
            explainer = data["explainer"]
            sv_single = explainer(input_scaled)
            if len(sv_single.shape) == 3:
                sv_single = sv_single[:, :, 1]
            fig_s, ax_s = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(sv_single[0], show=False)
            st.pyplot(plt.gcf())
            plt.close("all")

# ── TAB 4 — DATA EXPLORER ──────────────────────────────────────────
with tab4:
    st.subheader("Dataset Overview")
    df_clean = artefacts["df_clean"]
    st.markdown(f"**Shape:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns  ·  "
                f"**Default rate:** {df_clean['default'].mean():.2%}")

    st.dataframe(df_clean.head(100), use_container_width=True, height=350)

    st.markdown("#### Target Distribution")
    fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
    counts = df_clean["default"].value_counts()
    ax_dist.bar(["No Default (0)", "Default (1)"], counts.values,
                color=["#4CAF50", "#F44336"], edgecolor="black")
    for i, v in enumerate(counts.values):
        ax_dist.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
    ax_dist.set_ylabel("Count")
    st.pyplot(fig_dist)
    plt.close(fig_dist)

    st.markdown("#### Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
    corr = df_clean.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5, ax=ax_corr)
    st.pyplot(fig_corr)
    plt.close(fig_corr)

# ── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Explainable AI for Credit Card Default Risk Prediction · "
           "Mrinal Batsa · Manipal University Jaipur · 2026")
