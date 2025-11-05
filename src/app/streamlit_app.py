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

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

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

st.title("Student Dropout Predictor")

FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Tuition fees up to date",
]

ranges = {f: get_ranges(raw_df, f) for f in COMPACT_FEATURES}

with st.form("very_minimal_form"):
    age = st.slider(
        "Age at enrollment",
        min_value=int(ranges["Age at enrollment"][0]),
        max_value=int(ranges["Age at enrollment"][1]),
        value=int(ranges["Age at enrollment"][2]),
    )
    adm = st.slider(
        "Admission grade",
        float(ranges["Admission grade"][0]),
        float(ranges["Admission grade"][1]),
        float(ranges["Admission grade"][2]),
    )
    cu1 = st.number_input(
        "Curricular units 1st sem (approved)",
        min_value=int(ranges["Curricular units 1st sem (approved)"][0]),
        max_value=int(ranges["Curricular units 1st sem (approved)"][1]),
        value=int(ranges["Curricular units 1st sem (approved)"][2]),
    )
    tuition = st.selectbox(
        "Tuition fees up to date",
        options=sorted(raw_df["Tuition fees up to date"].dropna().unique() if raw_df is not None and "Tuition fees up to date" in raw_df.columns else ["Yes", "No"]),
    )
    submit = st.form_submit_button("Predict")

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
        st.warning("Model or preprocessor not found. Run training first: python src/models/train_models.py")
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
            st.metric("Dropout probability", f"{prob:.2%}" if prob is not None else "N/A")
            if pred == 1:
                st.error("Prediction: Dropout")
            else:
                st.success("Prediction: No Dropout")
        except Exception:
            st.error("Unable to preprocess input — please ensure the dataset and preprocessor were created by the project's pipeline.")

# --- Model performance section ---
st.markdown("---")
st.header("Model performance")
if metrics is None:
    st.info("No saved model metrics found. Run training (`python src/models/train_models.py`) to produce metrics in `outputs/model_results.json`.")
else:
    # metrics expected as dict: model_name -> { 'f1':..., 'report': {...} }
    rows = []
    for name, m in metrics.items():
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
        rows.append({"model": name, "accuracy": acc, "f1": f1, "precision_pos": prec1, "recall_pos": rec1})
    import pandas as _pd
    df_metrics = _pd.DataFrame(rows).set_index("model")
    st.table(df_metrics)
