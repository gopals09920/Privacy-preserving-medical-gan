import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ctgan import CTGAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Privacy-Preserving Synthetic Data Generator", layout="wide")

st.title("🏥 Synthetic Medical Data Generation with Privacy-Preserving CTGAN")
st.markdown("Generate synthetic medical records with statistical validation and privacy evaluation.")

# Sidebar Controls
st.sidebar.header("Pipeline Configuration")
epochs = st.sidebar.slider("CTGAN Epochs", min_value=50, max_value=500, value=100, step=50)
num_samples = st.sidebar.number_input("Number of Synthetic Samples", min_value=100, max_value=5000, value=1000, step=100)

# Load Baseline Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("clean_medical_data.csv")
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("`clean_medical_data.csv` nahi mili! Pehle data_prep.py run karein.")
else:
    st.subheader("1. Baseline Dataset Preview")
    st.dataframe(df.head())
    
    if st.button("🚀 Run Full Pipeline (Train GAN -> Validate -> Test Utility)"):
        categorical_cols = ['sex', 'cp', 'target']
        
        # 1. CTGAN Training
        with st.spinner("CTGAN Model Train ho raha hai..."):
            ctgan = CTGAN(epochs=epochs, batch_size=50, verbose=False)
            ctgan.fit(df, discrete_columns=categorical_cols)
            synth_df = ctgan.sample(num_samples)
            synth_df.to_csv("synthetic_medical_data.csv", index=False)
        st.success("✅ Synthetic Data Generation Complete!")
        
        # Download Button
        csv = synth_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Synthetic CSV", data=csv, file_name="synthetic_medical_data.csv", mime="text/csv")
        
        st.markdown("---")
        
        # 2. Statistical Validation
        st.subheader("2. Statistical Validation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlation Matrix Comparison**")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=axes[0])
            axes[0].set_title("Real Data")
            sns.heatmap(synth_df.corr(), annot=False, cmap="coolwarm", ax=axes[1])
            axes[1].set_title("Synthetic Data")
            st.pyplot(fig)
            
        with col2:
            st.write("**Age Distribution Comparison**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.kdeplot(df['age'], label='Real Data', fill=True, ax=ax2)
            sns.kdeplot(synth_df['age'], label='Synthetic Data', fill=True, ax=ax2)
            plt.legend()
            st.pyplot(fig2)
            
        st.markdown("---")
        
        # 3. Privacy Assessment (DCR)
        st.subheader("3. Privacy Evaluation (Distance to Closest Record)")
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(df)
        synth_scaled = scaler.transform(synth_df)
        
        distances = cdist(synth_scaled, real_scaled, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        exact_matches = np.sum(min_distances == 0)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Average DCR", f"{np.mean(min_distances):.4f}")
        m2.metric("Minimum DCR", f"{np.min(min_distances):.4f}")
        m3.metric("Exact Copies Leaked", exact_matches)
        
        if exact_matches == 0:
            st.success("🔒 Privacy Preserved: Zero exact copies found.")
        else:
            st.warning("⚠️ Warning: Data leakage detected.")
            
        st.markdown("---")
        
        # 4. Downstream ML Utility
        st.subheader("4. Downstream Machine Learning Utility")
        X_real, y_real = df.drop('target', axis=1), df['target']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
        
        X_synth, y_synth = synth_df.drop('target', axis=1), synth_df['target']
        
        rf_real = RandomForestClassifier(random_state=42).fit(X_train_r, y_train_r)
        rf_synth = RandomForestClassifier(random_state=42).fit(X_synth, y_synth)
        
        acc_real = accuracy_score(y_test_r, rf_real.predict(X_test_r))
        acc_synth = accuracy_score(y_test_r, rf_synth.predict(X_test_r))
        
        u1, u2 = st.columns(2)
        u1.metric("Model Trained on REAL Data Accuracy", f"{acc_real * 100:.2f}%")
        u2.metric("Model Trained on SYNTHETIC Data Accuracy", f"{acc_synth * 100:.2f}%")