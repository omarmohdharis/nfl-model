"""
08_export_for_explainer.py
--------------------------
Export the model outputs to a single JSON file that the
interactive explainer can load. Run this AFTER scripts 02-05
have produced their CSVs.
"""

import pandas as pd
import json
import os

V = pd.read_csv('output/value_function.csv')
fg = pd.read_csv('output/fg_by_yardline.csv')
punt = pd.read_csv('output/punt_values.csv')
conv = pd.read_csv('output/conversion_probability.csv')

data = {
    'V': V.set_index('own_yardline')['V'].to_dict(),
    'fg': fg.set_index('own_yardline')['p_make'].to_dict(),
    'punt': punt.set_index('own_yardline_kicking')['E_V_punt'].to_dict(),
    'conv': [
        {'ytg': int(r['ydstogo']), 'yl': int(r['own_yardline']), 'p': r['p_convert']}
        for _, r in conv.iterrows()
    ],
}

os.makedirs('explainer', exist_ok=True)
with open('explainer/model_data.json', 'w') as f:
    json.dump(data, f)

print(f"Exported model data to explainer/model_data.json")
print(f"  V function: {len(data['V'])} yard lines")
print(f"  FG probabilities: {len(data['fg'])} yard lines")
print(f"  Punt values: {len(data['punt'])} yard lines")
print(f"  Conversion probabilities: {len(data['conv'])} pairs")