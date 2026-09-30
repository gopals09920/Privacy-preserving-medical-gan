import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
print("--- DataSet Loading & Preprocessing ---")
try:
    heart_data = fetch_openml(name = 'heart-disease-uci', version = 1, as_frame = True)
    df = heart_data.frame
    print("Dataset successfully downloaded from OpenML!.")
except Exception as e:
    print("Online fetch issue, generating sample structured medical dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(29, 78, 500),
        'sex': np.random.choice([0, 1], 500),
        'cp': np.random.choice([0, 1, 2, 3], 500),
        'trestbps': np.random.randint(94, 200, 500),
        'chol': np.random.randint(126, 564, 500),
        'target': np.random.choice([0, 1], 500)
    })

print("\nDataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
categorical_cols = [col for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target'] if col in df.columns]
df = df.dropna()
print("\nCategorical Columns identified for CTGAN:", categorical_cols)
df.to_csv("clean_medical_data.csv", index=False)
print("\nClean Data saved to 'clean_medical_data.csv'!.")