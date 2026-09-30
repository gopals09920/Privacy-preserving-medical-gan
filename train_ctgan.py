import pandas as pd
from ctgan import CTGAN
print("--- CTGAN Model Training & Synthetic Data Generation ---")
df = pd.read_csv("clean_medical_data.csv")
categorical_cols = ['sex', 'cp', 'target']
print("Training CTGAN model.....")
ctgan = CTGAN(epochs = 100, batch_size = 50, verbose = True)
ctgan.fit(df, discrete_columns = categorical_cols)
print("\nTraining completed successfully!")
print("Generating 1000 synthetic medica records...")
synthetic_df = ctgan.sample(1000)
print("\nSynthetic Data Preview:")
print(synthetic_df.head())
synthetic_df.to_csv("synthetic_medical_data.csv", index = False)
print("\nSynthetic data saved to 'synthetic_medical_data.csv'!")