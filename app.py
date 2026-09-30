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

# Page Configuration
st.set_page_config(
    page_title="MedSynth AI | Privacy-Preserving GAN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Gradient UI)
st.markdown("""
<style>
    /* Dark Theme Accent Adjustments */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Metric Card Customizations */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Section Divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown('<div class="main-header">🧬 MedSynth AI Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Privacy-Preserving Synthetic Medical Data Generation & Privacy Auditing System</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/isometric/100/health-graph.png", width=70)
st.sidebar.title("⚙️ Engine Control")
st.sidebar.markdown("---")
epochs = st.sidebar.slider("CTGAN Training Epochs", min_value=50, max_value=500, value=100, step=50)
num_samples = st.sidebar.number_input("Synthetic Samples Count", min_value=100, max_value=5000, value=1000, step=100)

st.sidebar.info("💡 **Pro-Tip**: Higher epochs improve statistical utility but take longer to train.")

# Load Baseline Data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("clean_medical_data.csv")
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("`clean_medical_data.csv` not found! Please run `data_prep.py` first.")
else:
    # 1. Dataset Preview Section
    with st.container():
        st.markdown('### 📊 Baseline Clinical Dataset')
        st.dataframe(df.head(6), use_container_width=True)
    
    st.markdown("---")
    
    # Run Button
    col_btn, _ = st.columns([2, 3])
    with col_btn:
        run_pipeline = st.button("🚀 Execute End-to-End Synthesizer Pipeline", use_container_width=True)

    if run_pipeline:
        categorical_cols = ['sex', 'cp', 'target']
        
        # Training Phase
        with st.status("🔮 Synthesizing Medical Records...", expanded=True) as status:
            st.write("Initializing CTGAN model architecture...")
            ctgan = CTGAN(epochs=epochs, batch_size=50, verbose=False)
            
            st.write("Fitting conditional data distributions...")
            ctgan.fit(df, discrete_columns=categorical_cols)
            
            st.write("Sampling synthetic cohort...")
            synth_df = ctgan.sample(num_samples)
            synth_df.to_csv("synthetic_medical_data.csv", index=False)
            
            status.update(label="✅ Synthetic Pipeline Complete!", state="complete", expanded=False)

        # Download Section
        csv = synth_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Synthetic Data (CSV)",
            data=csv,
            file_name="synthetic_medical_data.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # 2. Validation Section
        st.markdown("### 📈 Statistical Distribution Fidelity")
        col1, col2 = st.columns(2)
        
        # Dark Theme Matplotlib Style
        plt.style.use('dark_background')
        
        with col1:
            st.markdown("##### Feature Correlation Dynamics")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.patch.set_facecolor('#0f172a')
            
            sns.heatmap(df.corr(), annot=False, cmap="mako", ax=axes[0], cbar=False)
            axes[0].set_title("Real Cohort", color='#94a3b8')
            axes[0].set_facecolor('#0f172a')
            
            sns.heatmap(synth_df.corr(), annot=False, cmap="mako", ax=axes[1], cbar=False)
            axes[1].set_title("Synthetic Cohort", color='#94a3b8')
            axes[1].set_facecolor('#0f172a')
            
            st.pyplot(fig)
            
        with col2:
            st.markdown("##### Age Demographics Comparison")
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            fig2.patch.set_facecolor('#0f172a')
            ax2.set_facecolor('#0f172a')
            
            sns.kdeplot(df['age'], label='Real Data', fill=True, color='#38bdf8', ax=ax2)
            sns.kdeplot(synth_df['age'], label='Synthetic Data', fill=True, color='#c084fc', ax=ax2)
            ax2.legend(facecolor='#1e293b', edgecolor='none')
            st.pyplot(fig2)
            
        st.markdown("---")
        
        # 3. Privacy Assessment (DCR)
        st.markdown("### 🔒 Privacy Audit & Risk Assessment (DCR)")
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(df)
        synth_scaled = scaler.transform(synth_df)
        
        distances = cdist(synth_scaled, real_scaled, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        exact_matches = np.sum(min_distances == 0)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Average DCR Score", f"{np.mean(min_distances):.4f}")
        m2.metric("Minimum DCR Score", f"{np.min(min_distances):.4f}")
        m3.metric("Exact Replications", exact_matches)
        
        if exact_matches == 0:
            st.success("🛡️ **Zero Privacy Leakage Detected**: Synthetic dataset contains no exact matches from the training baseline.")
        else:
            st.warning("⚠️ **Potential Leakage**: Replicated records identified. Consider tuning hyperparameters.")
            
        st.markdown("---")
        
        # 4. ML Utility
        st.markdown("### ⚡ Machine Learning Utility Verification")
        X_real, y_real = df.drop('target', axis=1), df['target']
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
        
        X_synth, y_synth = synth_df.drop('target', axis=1), synth_df['target']
        
        rf_real = RandomForestClassifier(random_state=42).fit(X_train_r, y_train_r)
        rf_synth = RandomForestClassifier(random_state=42).fit(X_synth, y_synth)
        
        acc_real = accuracy_score(y_test_r, rf_real.predict(X_test_r))
        acc_synth = accuracy_score(y_test_r, rf_synth.predict(X_test_r))
        
        u1, u2 = st.columns(2)
        u1.metric("Real Data Trained Accuracy", f"{acc_real * 100:.2f}%")
        u2.metric("Synthetic Data Trained Accuracy", f"{acc_synth * 100:.2f}%", delta=f"{(acc_synth - acc_real)*100:.2f}%")