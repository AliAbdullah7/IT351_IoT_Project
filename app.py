# -*- coding: utf-8 -*-

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="IoT Sensor Data Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        color: #6c757d;
        font-size: 18px;
        margin-top: 0px;
    }
    .section-note {
        background-color: #f8f9fa;
        padding: 14px 18px;
        border-radius: 10px;
        border-left: 5px solid #4e79a7;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
@st.cache_data
def load_data():
    db_path = Path("weather.db")
    if not db_path.exists():
        st.error("weather.db was not found. Please upload weather.db to the same GitHub repository as app.py.")
        st.stop()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sensor_readings", conn)
    conn.close()
    return df


@st.cache_resource
def train_model(data):
    features = ["humidity", "hour", "day", "month", "day_of_week"]
    target = "temperature"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=None,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    results = pd.DataFrame({
        "Actual Temperature": y_test.values,
        "Predicted Temperature": y_pred,
    })

    return model, X_test, y_test, y_pred, metrics, results


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
df = load_data()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek

model, X_test, y_test, y_pred, metrics, results = train_model(df)


# ------------------------------------------------------------
# Sidebar prediction controls
# ------------------------------------------------------------
st.sidebar.title("Prediction Controls")
st.sidebar.write("Enter sensor and time values to predict temperature.")

input_humidity = st.sidebar.number_input(
    "Humidity",
    min_value=float(df["humidity"].min()),
    max_value=float(df["humidity"].max()),
    value=float(round(df["humidity"].mean(), 2)),
    step=0.5,
)

input_hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=12, step=1)
input_day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15, step=1)
input_month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=8, step=1)
input_day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
)

input_data = pd.DataFrame({
    "humidity": [input_humidity],
    "hour": [input_hour],
    "day": [input_day],
    "month": [input_month],
    "day_of_week": [input_day_of_week],
})

predicted_temperature = model.predict(input_data)[0]

st.sidebar.success(f"Predicted Temperature: {predicted_temperature:.2f} C")


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown('<p class="main-title">IoT Sensor Data Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Interactive dashboard for analyzing IoT weather sensor data and predicting temperature using machine learning.</p>',
    unsafe_allow_html=True,
)

st.divider()

# Main KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Records", f"{df.shape[0]:,}")
with kpi2:
    st.metric("Features Used", "5")
with kpi3:
    st.metric("Sensor Type", str(df["sensor_type"].iloc[0]) if "sensor_type" in df.columns else "N/A")
with kpi4:
    st.metric("Predicted Temp", f"{predicted_temperature:.2f} C")

# ML metric cards
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("MAE", f"{metrics['MAE']:.3f}")
with m2:
    st.metric("RMSE", f"{metrics['RMSE']:.3f}")
with m3:
    st.metric("R2 Score", f"{metrics['R2']:.3f}")

st.divider()


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Overview",
    "Sensor Trends",
    "EDA Analysis",
    "ML Results",
])


# ------------------------------------------------------------
# Tab 1: Dataset overview
# ------------------------------------------------------------
with tab1:
    st.header("Dataset Overview")
    st.markdown(
        """
        <div class="section-note">
        The dataset contains IoT sensor readings collected from a DHT11 sensor. The main variables used in this project are timestamp, humidity, and temperature.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Sample Data")
        st.dataframe(df.head(15), use_container_width=True)

    with right:
        st.subheader("Missing Values")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df[["humidity", "temperature", "hour", "day", "month", "day_of_week"]].describe(), use_container_width=True)


# ------------------------------------------------------------
# Tab 2: Sensor trends
# ------------------------------------------------------------
with tab2:
    st.header("Sensor Data Trends")
    st.markdown(
        """
        <div class="section-note">
        These charts show how temperature and humidity changed over time. This helps understand the general behavior of the IoT sensor readings.
        </div>
        """,
        unsafe_allow_html=True,
    )

    trend_option = st.radio(
        "Choose trend chart",
        ["Temperature", "Humidity", "Both"],
        horizontal=True,
    )

    if trend_option in ["Temperature", "Both"]:
        st.subheader("Temperature Trend Over Time")
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df["timestamp"], df["temperature"])
        ax1.set_title("Temperature Trend Over Time")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Temperature")
        ax1.tick_params(axis="x", rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)

    if trend_option in ["Humidity", "Both"]:
        st.subheader("Humidity Trend Over Time")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df["timestamp"], df["humidity"])
        ax2.set_title("Humidity Trend Over Time")
        ax2.set_xlabel("Timestamp")
        ax2.set_ylabel("Humidity")
        ax2.tick_params(axis="x", rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)


# ------------------------------------------------------------
# Tab 3: EDA analysis
# ------------------------------------------------------------
with tab3:
    st.header("Exploratory Data Analysis")
    st.markdown(
        """
        <div class="section-note">
        EDA is used to understand relationships in the data before building the prediction model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Humidity vs Temperature")
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.scatter(df["humidity"], df["temperature"], alpha=0.7)
        ax3.set_title("Relationship Between Humidity and Temperature")
        ax3.set_xlabel("Humidity")
        ax3.set_ylabel("Temperature")
        fig3.tight_layout()
        st.pyplot(fig3)

    with col_b:
        st.subheader("Correlation Heatmap")
        corr_columns = ["humidity", "temperature", "hour", "day", "month", "day_of_week"]
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        sns.heatmap(df[corr_columns].corr(), annot=True, cmap="Blues", ax=ax4)
        ax4.set_title("Correlation Between Features")
        fig4.tight_layout()
        st.pyplot(fig4)

    st.info(
        "The correlation results show that humidity and time-based features are useful for predicting temperature."
    )


# ------------------------------------------------------------
# Tab 4: ML results
# ------------------------------------------------------------
with tab4:
    st.header("Machine Learning Model Results")
    st.markdown(
        """
        <div class="section-note">
        A Random Forest Regressor model was trained to predict temperature using humidity and time-based features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Model Inputs and Target")
    st.write("Input features: humidity, hour, day, month, day_of_week")
    st.write("Prediction target: temperature")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean Absolute Error", f"{metrics['MAE']:.3f}")
    with c2:
        st.metric("Root Mean Squared Error", f"{metrics['RMSE']:.3f}")
    with c3:
        st.metric("R2 Score", f"{metrics['R2']:.3f}")

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("Actual vs Predicted Temperature")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.scatter(y_test, y_pred, alpha=0.7)
        min_value = min(y_test.min(), y_pred.min())
        max_value = max(y_test.max(), y_pred.max())
        ax5.plot([min_value, max_value], [min_value, max_value], linestyle="--")
        ax5.set_title("Actual vs Predicted Temperature")
        ax5.set_xlabel("Actual Temperature")
        ax5.set_ylabel("Predicted Temperature")
        fig5.tight_layout()
        st.pyplot(fig5)

    with right:
        st.subheader("Prediction Sample")
        st.dataframe(results.head(20), use_container_width=True)

    st.success(
        "The model achieved strong performance because the prediction error is low and the R2 score is high."
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()
st.caption("IT351 - Data Science for the Internet of Things | IoT Sensor Data Prediction Dashboard")
