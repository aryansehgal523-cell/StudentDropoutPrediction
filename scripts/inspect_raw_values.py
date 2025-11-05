import pandas as pd
from pathlib import Path
ROOT=Path('/workspace/StudentDropoutPrediction')
raw=ROOT/'data'/'raw'/'data.csv'
try:
    df=pd.read_csv(raw, sep=';', engine='python')
except Exception:
    df=pd.read_csv(raw)
df.columns=df.columns.str.strip()
cols=['Tuition fees up to date','Gender','Scholarship holder','Debtor']
for c in cols:
    if c in df.columns:
        print('\nColumn:',c,'unique values:')
        print(df[c].dropna().unique()[:50])
    else:
        print('\nColumn',c,'not found')
