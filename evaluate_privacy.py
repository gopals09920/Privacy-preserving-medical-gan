import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
print("--- Privacy Risk Evaluation (DCR) ---")
real_df = pd.read_csv("clean_medical_data.csv")
synth_df = pd.read_csv("synthetic_medical_data.csv")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
real_scaled = scaler.fit_transform(real_df)
synth_scaled = scaler.transform(synth_df)
distances = cdist(synth_scaled, real_scaled, metric='euclidean')
min_distances = np.min(distances, axis = 1)
print(f"Average DCR: {np.min(min_distances):.4f}")
print(f"Minimum DCR: {np.min(min_distances):.4f}")
exact_matches = np.sum(min_distances == 0)
print(f"Exact copies count: {exact_matches}")
if exact_matches == 0:
    print("\nSUCCESS: koi real patient data leak nahi hua (Zero exact copies)!")
else:
    print("\nWarning kuch synthetic records exact real data jaise hain.")
