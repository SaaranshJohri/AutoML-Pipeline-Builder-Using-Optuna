import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import joblib
import optuna.visualization as vis
from src.optimizer import run_optimization
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AutoML Pipeline Builder", layout="wide")

st.title("🚀 AutoML Pipeline Builder")

# Upload
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        df.columns = df.columns.str.strip().str.lower()
        target = target.lower()

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        X = df.drop(columns=[target])
        y = df[target]

        # Encode target if categorical
        if y.dtype == "object" or str(y.dtype) == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)

        with st.spinner("Running AutoML Optimization..."):
            study = run_optimization(X, y, n_trials=20)

        st.success("Optimization Complete!")

        # Results
        st.subheader("🏆 Best Results")
        st.write("Best Params:", study.best_trial.params)
        st.write("Best Score:", study.best_value)

        # Save model (placeholder)
        joblib.dump(study.best_trial.params, "best_model.pkl")

        # Download
        with open("best_model.pkl", "rb") as f:
            st.download_button("📥 Download Best Model Params", f, file_name="best_model.pkl")

        # Charts
        st.subheader("📊 Optimization History")
        st.plotly_chart(vis.plot_optimization_history(study))

        st.subheader("⚙️ Parameter Importance")
        st.plotly_chart(vis.plot_param_importances(study))

        st.subheader("🔗 Parallel Coordinate Plot")
        st.plotly_chart(vis.plot_parallel_coordinate(study))