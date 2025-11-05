from pathlib import Path
import json
import joblib

ROOT = Path('/workspace/StudentDropoutPrediction')
P = ROOT / 'outputs' / 'model_results.json'

def try_load(path):
    # try json then joblib
    try:
        with open(path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        try:
            return joblib.load(path)
        except Exception as e:
            print('Could not load metrics file as json or joblib:', e)
            return None


data = try_load(P)
if data is None:
    print('No readable metrics found at', P)
else:
    print('Models and metrics found in', P)
    # data may be dict mapping model_name -> metrics dict
    if isinstance(data, dict):
        for name, metrics in data.items():
            print('\nModel:', name)
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    print(f'  {k}: {v}')
            else:
                print(' ', metrics)
    else:
        print(data)

