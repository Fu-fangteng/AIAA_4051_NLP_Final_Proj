import pickle
from pathlib import Path
import numpy as np

pkl_path = Path('aiaa4051/task3/comparison/param_relevance_data.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print('=== Data Summary ===')
print(f'Layers: {data["layers"]}')
print()
print('Model A (SciQ) relevance:')
for layer, val in zip(data['layers'], data['rel_A_norm']):
    print(f'  L{int(layer)}: {val:.4f}')
print()
print('Model B (SQuAD) relevance:')
for layer, val in zip(data['layers'], data['rel_B_norm']):
    print(f'  L{int(layer)}: {val:.4f}')
print()
print('Difference (SciQ - SQuAD):')
for layer, val in zip(data['layers'], data['diff_norm']):
    print(f'  L{int(layer)}: {val:+.4f}')

print()
print('=== Statistics ===')
print(f'SciQ - Max: {np.max(data["rel_A_norm"]):.4f}, Mean: {np.mean(data["rel_A_norm"]):.4f}')
print(f'SQuAD - Max: {np.max(data["rel_B_norm"]):.4f}, Mean: {np.mean(data["rel_B_norm"]):.4f}')
print(f'Max difference: {np.max(np.abs(data["diff_norm"])):.4f}')
corr = np.corrcoef(data["rel_A_norm"], data["rel_B_norm"])[0,1]
print(f'Correlation: {corr:.4f}')
