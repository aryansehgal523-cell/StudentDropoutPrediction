"""
Streamlit app for predicting student dropout and viewing analytics.
Run with:
    streamlit run src/app/streamlit_app.py
"""
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "outputs"
PROC = BASE / "data" / "processed"
MODEL_PATH = OUT / "best_model.joblib"
PREPROCESSOR_PATH = PROC / "preprocessor.joblib"
FEATURES_PATH = PROC / "feature_names.csv"

st.set_page_config(
    page_title="🎓 Student Dropout Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Student Dropout Prediction ML App\nBuilt with Streamlit, scikit-learn, and XGBoost"
    }
)

# Custom CSS for vibrant, attractive UI
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem !important;
    }
    
    /* Section headers */
    h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Form container with glass effect */
    .stForm {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
        backdrop-filter: blur(4px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
    }
    
    /* Input labels */
    .stForm label {
        color: #2d3748 !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Metrics container */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        border-radius: 15px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
        color: #065f46 !important;
        font-weight: 600 !important;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
        color: #7f1d1d !important;
        font-weight: 600 !important;
    }
    
    /* Tables */
    table {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    
    thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem !important;
    }
    
    tbody tr:hover {
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Card effect for content */
    .element-container {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Use the project root detected from this file to locate data/artifacts so the
# app works both in local clones and different runtime containers.
ROOT = BASE
RAW_CSV = ROOT / "data" / "raw" / "data.csv"
# prefer the earlier defined MODEL_PATH / PREPROCESSOR_PATH if present, but
# make sure paths are consistently relative to the discovered ROOT
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
PREPROCESSOR_PATH = ROOT / "data" / "processed" / "preprocessor.joblib"
RESULTS_JSON = ROOT / "outputs" / "model_results.json"

@st.cache_data(show_spinner=False)
def load_raw_data():
    if RAW_CSV.exists():
        try:
            df = pd.read_csv(RAW_CSV, sep=";", engine="python")
        except Exception:
            df = pd.read_csv(RAW_CSV)
        df.columns = df.columns.str.strip()
        return df
    return None

@st.cache_resource
def load_artifacts():
    model = None
    preprocessor = None
    metrics = None
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    if PREPROCESSOR_PATH.exists():
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        except Exception:
            preprocessor = None
    if RESULTS_JSON.exists():
        # metrics may be saved either as JSON or as a joblib/pickle object
        try:
            import json

            with open(RESULTS_JSON, "r", encoding="utf8") as f:
                metrics = json.load(f)
        except Exception:
            # fallback: try loading via joblib in case the file is a pickled dict
            try:
                metrics = joblib.load(RESULTS_JSON)
            except Exception:
                metrics = None
    return model, preprocessor, metrics

raw_df = load_raw_data()
model, preprocessor, metrics = load_artifacts()

COMPACT_FEATURES = [
    "Age at enrollment",
    "Gender",
    "Admission grade",
    "Tuition fees up to date",
    "Scholarship holder",
    "Debtor",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "GDP",
]

NUMERIC_COLS = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "GDP",
]

def get_ranges(df, feature):
    if df is None or feature not in df.columns:
        # sensible defaults
        defaults = {
            "Age at enrollment": (17, 60, 22),
            "Admission grade": (0.0, 20.0, 12.0),
            "Curricular units 1st sem (approved)": (0, 30, 5),
            "Curricular units 1st sem (grade)": (0.0, 20.0, 12.0),
            "Curricular units 2nd sem (approved)": (0, 30, 4),
            "Curricular units 2nd sem (grade)": (0.0, 20.0, 11.0),
            "Unemployment rate": (0.0, 100.0, 6.5),
            "GDP": (0.0, 1_000_000.0, 20000.0),
        }
        return defaults.get(feature, (0, 1, 0))
    ser = pd.to_numeric(df[feature], errors="coerce")
    mn = float(np.nanmin(ser)) if not ser.dropna().empty else 0.0
    mx = float(np.nanmax(ser)) if not ser.dropna().empty else mn + 1.0
    med = float(np.nanmedian(ser.dropna())) if not ser.dropna().empty else (mn + mx) / 2.0
    # clamp
    if mn == mx:
        mx = mn + 1.0
    return mn, mx, med

def coerce_input_df(df):
    # Ensure numeric cols are numeric (coerce errors to NaN so imputer can handle)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep only compact features and order them
    df = df.reindex(columns=COMPACT_FEATURES)
    return df

COMPACT_FEATURES = COMPACT_FEATURES

IMPORTANT_FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Tuition fees up to date",
]

st.markdown("<h1>🎓 Student Dropout Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem; margin-bottom: 2rem;'>Predict student dropout risk using advanced machine learning models</p>", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Tuition fees up to date",
]

ranges = {f: get_ranges(raw_df, f) for f in COMPACT_FEATURES}

with col2:
    with st.form("very_minimal_form"):
        st.markdown("### 📝 Student Information")
        age = st.slider(
            "👤 Age at enrollment",
            min_value=int(ranges["Age at enrollment"][0]),
            max_value=int(ranges["Age at enrollment"][1]),
            value=int(ranges["Age at enrollment"][2]),
        )
        adm = st.slider(
            "📊 Admission grade",
            float(ranges["Admission grade"][0]),
            float(ranges["Admission grade"][1]),
            float(ranges["Admission grade"][2]),
        )
        cu1 = st.number_input(
            "✅ Curricular units 1st sem (approved)",
            min_value=int(ranges["Curricular units 1st sem (approved)"][0]),
            max_value=int(ranges["Curricular units 1st sem (approved)"][1]),
            value=int(ranges["Curricular units 1st sem (approved)"][2]),
        )
        tuition = st.selectbox(
            "💰 Tuition fees up to date",
            options=sorted(raw_df["Tuition fees up to date"].dropna().unique() if raw_df is not None and "Tuition fees up to date" in raw_df.columns else ["Yes", "No"]),
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Predict Dropout Risk", width="stretch")

if submit:
    inp = {
        "Age at enrollment": age,
        "Admission grade": adm,
        "Curricular units 1st sem (approved)": cu1,
        "Tuition fees up to date": tuition,
    }
    df_in = pd.DataFrame([inp])
    # coerce numeric fields
    for c in ["Age at enrollment", "Admission grade", "Curricular units 1st sem (approved)"]:
        df_in[c] = pd.to_numeric(df_in[c], errors="coerce")

    if model is None or preprocessor is None:
        st.warning("⚠️ Model or preprocessor not found. Run training first: `python src/models/train_models.py`")
    else:
        try:
            # Ensure the input contains the exact columns the preprocessor was fitted on.
            try:
                cols_in = list(preprocessor.feature_names_in_)
            except Exception:
                # fallback: use our COMPACT_FEATURES as order
                cols_in = COMPACT_FEATURES
            # add missing columns with NaN so imputer can fill them
            for c in cols_in:
                if c not in df_in.columns:
                    df_in[c] = np.nan
            # normalize common yes/no strings to numeric 1/0 (training data uses 0/1)
            yes_vals = {"yes", "y", "true", "1"}
            no_vals = {"no", "n", "false", "0"}
            for c in df_in.columns:
                if df_in[c].dtype == object:
                    # map case-insensitive for non-null values only
                    mapped = df_in[c].astype(str).str.strip().str.lower()
                    non_null = mapped[~mapped.isin(["nan", "none", "nan.0"])].dropna()
                    if not non_null.empty and non_null.isin(yes_vals | no_vals).all():
                        df_in[c] = mapped.apply(lambda v: 1 if v in yes_vals else 0)
            # coerce numeric-like columns to numeric so imputers can work
            for c in cols_in:
                df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
            # reorder
            df_in = df_in[cols_in]

            Xp = preprocessor.transform(df_in)
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(Xp)[:, 1][0]
            pred = int(model.predict(Xp)[0])
            
            # Display results in centered column with nice formatting
            st.markdown("<br>", unsafe_allow_html=True)
            with col2:
                if prob is not None:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("📊 Dropout Probability", f"{prob:.1%}")
                    with col_b:
                        st.metric("🎯 Risk Level", "High" if prob > 0.5 else "Low")
                
                if pred == 1:
                    st.error("⚠️ **Prediction: Student at Risk of Dropout**")
                    st.markdown("""
                    <div style='background: rgba(254,202,202,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <p style='color: white; margin: 0;'>
                            <strong>Recommendation:</strong> Consider providing additional support such as tutoring, 
                            counseling, or financial assistance to help retain this student.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ **Prediction: Student Expected to Continue**")
                    st.markdown("""
                    <div style='background: rgba(167,243,208,0.2); padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <p style='color: white; margin: 0;'>
                            <strong>Status:</strong> This student shows positive indicators for academic continuation. 
                            Continue monitoring progress and maintain current support levels.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception as e:
            import traceback
            st.error(f"❌ Unable to process prediction: {str(e)}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
                st.info("💡 **Tip**: Try retraining the models with: `python src/models/train_models.py`")

# --- Model performance section ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>📈 Model Performance Metrics</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; margin-bottom: 2rem;'>Comparison of different machine learning models</p>", unsafe_allow_html=True)

if metrics is None:
    st.info("ℹ️ No saved model metrics found. Run training (`python src/models/train_models.py`) to produce metrics in `outputs/model_results.json`.")
else:
    # metrics expected as dict: model_name -> { 'f1':..., 'report': {...} }
    # map short keys to readable names (ensure xgb shows as XGBoost)
    pretty = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
    rows = []
    for name, m in metrics.items():
        display_name = pretty.get(name, name)
        report = m.get("report") if isinstance(m, dict) else None
        f1 = m.get("f1") if isinstance(m, dict) else None
        acc = None
        prec1 = None
        rec1 = None
        if report and isinstance(report, dict):
            try:
                acc = report.get("accuracy")
            except Exception:
                acc = None
            # class '1' metrics may be keyed by string
            cls1 = report.get("1") or report.get(1)
            if isinstance(cls1, dict):
                prec1 = cls1.get("precision")
                rec1 = cls1.get("recall")
        # format numeric values for readability
        def fmt(x):
            try:
                return round(float(x), 3)
            except Exception:
                return x

        rows.append({"Model": display_name, "Accuracy": fmt(acc), "F1 Score": fmt(f1), "Precision": fmt(prec1), "Recall": fmt(rec1)})
    
    import pandas as _pd
    df_metrics = _pd.DataFrame(rows).set_index("Model")
    # sort by f1 descending if available
    if "F1 Score" in df_metrics.columns:
        df_metrics = df_metrics.sort_values(by="F1 Score", ascending=False)
    
    # Display in centered column
    col_left, col_center, col_right = st.columns([0.5, 2, 0.5])
    with col_center:
        st.dataframe(df_metrics, width="stretch")
        
        # Add a simple bar chart for F1 scores
        if "F1 Score" in df_metrics.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#667eea', '#764ba2', '#f093fb']
            df_metrics['F1 Score'].plot(kind='barh', ax=ax, color=colors[:len(df_metrics)])
            ax.set_xlabel('F1 Score', fontweight='bold', fontsize=12)
            ax.set_title('Model Comparison by F1 Score', fontweight='bold', fontsize=14, pad=20)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem;'>Built with ❤️ using Streamlit, scikit-learn, and XGBoost</p>", unsafe_allow_html=True)
