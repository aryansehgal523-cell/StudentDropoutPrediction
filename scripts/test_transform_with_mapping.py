from pathlib import Path
import pandas as pd, joblib, traceback, numpy as np
ROOT=Path('/workspace/StudentDropoutPrediction')
P=(ROOT/'data'/'processed'/'preprocessor.joblib')
pre=joblib.load(P)
print('Loaded preprocessor')
# sample input with 'Yes'
df=pd.DataFrame([{'Age at enrollment':20,'Admission grade':12.61,'Curricular units 1st sem (approved)':5,'Tuition fees up to date':'Yes'}])
print('before:', df.dtypes.to_dict(), df.iloc[0].to_dict())
# mimic app normalization
try:
    cols_in = list(pre.feature_names_in_)
except Exception:
    cols_in = ['Age at enrollment','Gender','Admission grade','Tuition fees up to date']
for c in cols_in:
    if c not in df.columns:
        df[c] = np.nan
# mapping
yes_vals = {"yes","y","true","1"}
no_vals = {"no","n","false","0"}
for c in df.columns:
    if df[c].dtype == object:
        mapped = df[c].astype(str).str.strip().str.lower()
        non_null = mapped[~mapped.isin(["nan","none","nan.0"])].dropna()
        print('col',c,'non_null:', non_null.tolist())
        if not non_null.empty and non_null.isin(yes_vals | no_vals).all():
            df[c] = mapped.apply(lambda v: 1 if v in yes_vals else 0)
# coerce
for c in cols_in:
    df[c] = pd.to_numeric(df[c], errors='coerce')
print('after:', df.dtypes.to_dict(), df.iloc[0].to_dict())
try:
    X=pre.transform(df)
    print('transform succeeded shape', X.shape)
except Exception:
    traceback.print_exc()
