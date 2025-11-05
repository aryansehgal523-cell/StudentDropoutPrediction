from pathlib import Path
import pandas as pd, joblib, traceback, numpy as np
ROOT=Path('/workspace/StudentDropoutPrediction')
P=(ROOT/'data'/'processed'/'preprocessor.joblib')
M=(ROOT/'outputs'/'best_model.joblib')
pre=joblib.load(P)
model=joblib.load(M)
print('Loaded preprocessor and model')
# sample input
df=pd.DataFrame([{'Age at enrollment':20,'Admission grade':12.61,'Curricular units 1st sem (approved)':5,'Tuition fees up to date':'Yes'}])
# normalization (same as app)
cols_in = list(pre.feature_names_in_)
for c in cols_in:
    if c not in df.columns:
        df[c] = np.nan
yes_vals = {"yes","y","true","1"}
no_vals = {"no","n","false","0"}
for c in df.columns:
    if df[c].dtype == object:
        mapped = df[c].astype(str).str.strip().str.lower()
        non_null = mapped[~mapped.isin(["nan","none","nan.0"])].dropna()
        if not non_null.empty and non_null.isin(yes_vals | no_vals).all():
            df[c] = mapped.apply(lambda v: 1 if v in yes_vals else 0)
for c in cols_in:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print('Prepared input:', df.iloc[0].to_dict())
X=pre.transform(df)
print('Transformed shape', X.shape)
if hasattr(model,'predict_proba'):
    print('prob', model.predict_proba(X)[:,1][0])
print('pred', model.predict(X)[0])
