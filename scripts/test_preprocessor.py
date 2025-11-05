from pathlib import Path
import pandas as pd, joblib, traceback
ROOT=Path('/workspace/StudentDropoutPrediction')
P=(ROOT/'data'/'processed'/'preprocessor.joblib')
print('preprocessor exists:', P.exists())
if not P.exists():
    raise SystemExit('No preprocessor')
pre=joblib.load(P)
print('Loaded preprocessor:', type(pre))
# Build sample input similar to app
inp={'Age at enrollment':20,'Admission grade':12.61,'Curricular units 1st sem (approved)':5,'Tuition fees up to date':'Yes'}
df=pd.DataFrame([inp])
print('Input df columns:', df.columns.tolist())
try:
    # mimic app: ensure columns expected by preprocessor
    try:
        cols = list(pre.feature_names_in_)
    except Exception:
        cols = ['Age at enrollment','Gender','Admission grade','Tuition fees up to date']
    for c in cols:
        if c not in df.columns:
            df[c] = float('nan')
    df = df[cols]
    X=pre.transform(df)
    print('Transform success, shape', X.shape)
except Exception as e:
    print('Transform failed')
    traceback.print_exc()
