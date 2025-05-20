import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    california = fetch_california_housing(as_frame=True)
    return california.frame

# Latih model regresi
@st.cache_data
def train_model(data):
    X = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return model, X.columns, mse, r2, data.corr(), y, y_pred

def main():
    st.set_page_config(page_title="Regresi Linear California Housing", layout="wide")
    st.title("🏡 Prediksi Median House Value di California")

    data = load_data()
    model, feature_names, mse, r2, corr_matrix, y_actual, y_pred = train_model(data)

    st.subheader("🔧 Input Fitur untuk Prediksi")
    input_data = []
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        col = col1 if i % 2 == 0 else col2
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())
        val = col.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
        input_data.append(val)

    if st.button("Prediksi Median House Value"):
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        st.success(f"🏠 Estimasi Median House Value: **${prediction * 100000:.2f}**")

    st.subheader("📊 Evaluasi Model")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    st.subheader("📈 Visualisasi")
    with st.expander("🔍 Heatmap Korelasi Fitur"):
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax1)
        st.pyplot(fig1)

    with st.expander("🎯 Grafik Aktual vs Prediksi"):
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=y_actual, y=y_pred, alpha=0.5, ax=ax2)
        ax2.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--')
        ax2.set_xlabel("Actual Median House Value")
        ax2.set_ylabel("Predicted Median House Value")
        ax2.set_title("Actual vs Predicted Median House Value")
        st.pyplot(fig2)

    with st.expander("📋 Tampilkan Data Mentah"):
        st.dataframe(data)

if __name__ == "__main__":
    main()
