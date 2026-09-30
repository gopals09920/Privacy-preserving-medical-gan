import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print("--- Downstream Machine Learning Utility Test ---")
real_df = pd.read_csv("clean_medical_data.csv")
synth_df = pd.read_csv("synthetic_medical_data.csv")
X_real = real_df.drop('target', axis = 1)
y_real = real_df['target']
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size = 0.2, random_state = 42
)
X_synth = synth_df.drop('target', axis = 1)
y_synth = synth_df['target']
rf_real = RandomForestClassifier(random_state = 42)
rf_real.fit(X_train_real, y_train_real)
acc_real = accuracy_score(y_test_real, rf_real.predict(X_test_real))
rf_synth = RandomForestClassifier(random_state = 42)
rf_synth.fit(X_synth, y_synth)
acc_synth = accuracy_score(y_test_real, rf_synth.predict(X_test_real))
print(f"\nModel Trained on REAL Data Accuracy: {acc_real * 100:.2f}%")
print(f"Model Trained on SYNTHETIC Data Accuracy: {acc_synth * 100:.2f}%")
if abs(acc_real - acc_synth) <= 0.15:
    print("\nSUCCESS: Synthetic data retains high Machine Learning Utility!")
else:
    print("\nNOTE: Gap in accuracy. CTGAN epochs can be increased for better utility.")