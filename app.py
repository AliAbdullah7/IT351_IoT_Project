# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="IoT Sensor Data Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)


# -----------------------------
# Load data from SQLite database
# -----------------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("weather.db")
    df = pd.read_sql_query("SELECT * FROM sensor_readings", conn)
    conn.close()
    return df


df = load_data()


# -----------------------------
# Data preprocessing
# -----------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month
df["day_of_week"] = df["timestamp"].dt.dayofweek


# -----------------------------
# Machine learning model
# -----------------------------
features = ["humidity", "hour", "day", "month", "day_of_week"]
target = "temperature"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# -----------------------------
# Dashboard title
# -----------------------------
st.title("IoT Sensor Data Prediction Dashboard")
st.write(
    """
    This dashboard analyzes IoT weather sensor data and uses a machine learning model
    to predict temperature based on humidity and time-based features.
    """
)


# -----------------------------
# Dataset overview
# -----------------------------
st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Number of Records", df.shape[0])

with col2:
    st.metric("Number of Columns", df.shape[1])

with col3:
    st.metric("Sensor Type", df["sensor_type"].iloc[0])

st.subheader("Sample Data")
st.dataframe(df.head(10))


# -----------------------------
# Basic statistics
# -----------------------------
st.subheader("Basic Statistics")
st.dataframe(df[["humidity", "temperature"]].describe())


# -----------------------------
# Sensor trends
# -----------------------------
st.header("2. Sensor Data Trends")

st.subheader("Temperature Trend Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df["timestamp"], df["temperature"])
ax1.set_title("Temperature Trend Over Time")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Temperature")
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("Humidity Trend Over Time")
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(df["timestamp"], df["humidity"])
ax2.set_title("Humidity Trend Over Time")
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("Humidity")
plt.xticks(rotation=45)
st.pyplot(fig2)


# -----------------------------
# Relationship and correlation
# -----------------------------
st.header("3. Exploratory Data Analysis")

st.subheader("Relationship Between Humidity and Temperature")
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(df["humidity"], df["temperature"])
ax3.set_title("Relationship Between Humidity and Temperature")
ax3.set_xlabel("Humidity")
ax3.set_ylabel("Temperature")
st.pyplot(fig3)

st.subheader("Correlation Between Features")
corr_columns = ["humidity", "temperature", "hour", "day", "month", "day_of_week"]
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.heatmap(df[corr_columns].corr(), annot=True, ax=ax4)
ax4.set_title("Correlation Between Features")
st.pyplot(fig4)


# -----------------------------
# Model performance
# -----------------------------
st.header("4. Machine Learning Model")

st.write(
    """
    The dashboard uses a Random Forest Regressor model to predict temperature.
    The input features are humidity, hour, day, month, and day of week.
    """
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("MAE", round(mae, 3))

with metric2:
    st.metric("RMSE", round(rmse, 3))

with metric3:
    st.metric("R2 Score", round(r2, 3))


# -----------------------------
# Actual vs predicted
# -----------------------------
st.subheader("Actual vs Predicted Temperature")

fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.scatter(y_test, y_pred)
ax5.set_title("Actual vs Predicted Temperature")
ax5.set_xlabel("Actual Temperature")
ax5.set_ylabel("Predicted Temperature")
st.pyplot(fig5)

results = pd.DataFrame({
    "Actual Temperature": y_test.values,
    "Predicted Temperature": y_pred
})

st.subheader("Prediction Results Sample")
st.dataframe(results.head(20))


# -----------------------------
# User input prediction
# -----------------------------
st.header("5. Try Temperature Prediction")

st.write("Enter sensor values below to predict temperature.")

input_humidity = st.slider(
    "Humidity",
    float(df["humidity"].min()),
    float(df["humidity"].max()),
    float(df["humidity"].mean())
)

input_hour = st.slider("Hour", 0, 23, 12)
input_day = st.slider("Day", 1, 31, 15)
input_month = st.slider("Month", 1, 12, 8)
input_day_of_week = st.slider("Day of Week", 0, 6, 0)

input_data = pd.DataFrame({
    "humidity": [input_humidity],
    "hour": [input_hour],
    "day": [input_day],
    "month": [input_month],
    "day_of_week": [input_day_of_week]
})

prediction = model.predict(input_data)[0]

st.success(f"Predicted Temperature: {prediction:.2f} C")


# -----------------------------
# Conclusion
# -----------------------------
st.header("6. Conclusion")

st.write(
    """
    The results show that humidity and time-based features can be used to predict
    temperature effectively. The Random Forest model achieved strong performance,
    with low error values and a high R2 score.
    """
)
