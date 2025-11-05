from pathlib import Path
import joblib
ROOT=Path('/workspace/StudentDropoutPrediction')
P=ROOT/'data'/'processed'/'preprocessor.joblib'
pre=joblib.load(P)
print('preprocessor type:', type(pre))
print('transformers attr present:', hasattr(pre, 'transformers'))
try:
    for t in pre.transformers:
        print('->', t[0], ' cols=', t[2])
except Exception as e:
    print('reading transformers failed', e)
try:
    print('feature_names_in_:', list(pre.feature_names_in_))
except Exception:
    print('no feature_names_in_')
# try to extract numeric/cat lists from attributes used during fitting
try:
    num = pre.transformers_[0][2]
    cat = pre.transformers_[1][2]
    print('transformers_ numeric cols:', num)
    print('transformers_ categorical cols:', cat)
except Exception as e:
    print('could not read transformers_', e)
