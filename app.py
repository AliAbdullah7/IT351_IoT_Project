# -*- coding: utf-8 -*-

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="IoT Sensor Data Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Styling
# =====================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }

    .section-card {
        background: rgba(120, 120, 120, 0.10);
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 12px 0 20px 0;
        line-height: 1.7;
    }

    .prediction-card {
        background: rgba(40, 167, 69, 0.18);
        border: 1px solid rgba(40, 167, 69, 0.45);
        border-radius: 16px;
        padding: 20px;
        margin-top: 14px;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .warning-card {
        background: rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.45);
        border-radius: 16px;
        padding: 18px;
        margin-top: 14px;
    }

    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.10);
        border: 1px solid rgba(120, 120, 120, 0.22);
        padding: 16px;
        border-radius: 16px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.25);
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
        height: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Load Dataset
# =====================================================
@st.cache_data
def load_data():
    db_path = Path("weather.db")

    if not db_path.exists():
        st.error("weather.db was not found. Please upload weather.db in the same GitHub repository as app.py.")
        st.stop()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sensor_readings", conn)
    conn.close()

    return df


# =====================================================
# Preprocessing
# =====================================================
@st.cache_data
def preprocess_data(df):
    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "humidity", "temperature"])

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_month"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    data = data.sort_values("timestamp")

    return data


# =====================================================
# Train ML Model
# =====================================================
@st.cache_resource
def train_model(data):
    features = ["humidity", "hour", "day_of_month", "month", "day_of_week"]
    target = "temperature"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "Actual Temperature": y_test.values,
        "Predicted Temperature": y_pred
    })

    results["Actual Temperature"] = results["Actual Temperature"].round(2)
    results["Predicted Temperature"] = results["Predicted Temperature"].round(2)

    return model, X_test, y_test, y_pred, mae, rmse, r2, results


# =====================================================
# Helper Functions
# =====================================================
def create_line_chart(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, y, linewidth=1.6)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return fig


def create_actual_vs_predicted_chart(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.scatter(y_test, y_pred, alpha=0.75)

    min_value = min(float(y_test.min()), float(y_pred.min()))
    max_value = max(float(y_test.max()), float(y_pred.max()))

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2
    )

    ax.set_title("Actual vs Predicted Temperature", fontsize=14)
    ax.set_xlabel("Actual Temperature (C)")
    ax.set_ylabel("Predicted Temperature (C)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# =====================================================
# Load and Prepare Data
# =====================================================
df_raw = load_data()
df = preprocess_data(df_raw)
model, X_test, y_test, y_pred, mae, rmse, r2, results = train_model(df)


# =====================================================
# Dictionaries
# =====================================================
month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}


# =====================================================
# Sidebar Prediction Section
# =====================================================
st.sidebar.title("Temperature Prediction")
st.sidebar.write("Enter the sensor and time values, then click Predict.")

humidity_input = st.sidebar.number_input(
    "Humidity",
    min_value=float(df["humidity"].min()),
    max_value=float(df["humidity"].max()),
    value=float(round(df["humidity"].mean(), 2)),
    step=0.5
)

hour_input = st.sidebar.selectbox(
    "Hour of Day",
    options=list(range(24)),
    index=12,
    format_func=lambda x: f"{x:02d}:00"
)

day_input = st.sidebar.number_input(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15,
    step=1
)

month_input = st.sidebar.selectbox(
    "Month",
    options=list(month_names.keys()),
    index=7,
    format_func=lambda x: month_names[x]
)

day_week_input = st.sidebar.selectbox(
    "Day of Week",
    options=list(day_names.keys()),
    index=0,
    format_func=lambda x: day_names[x]
)

predict_button = st.sidebar.button("Predict Temperature")

if predict_button:
    input_df = pd.DataFrame({
        "humidity": [humidity_input],
        "hour": [hour_input],
        "day_of_month": [day_input],
        "month": [month_input],
        "day_of_week": [day_week_input]
    })

    prediction = float(model.predict(input_df)[0])

    st.session_state["prediction"] = prediction
    st.session_state["inputs"] = {
        "Humidity": humidity_input,
        "Hour": f"{hour_input:02d}:00",
        "Day of Month": day_input,
        "Month": month_names[month_input],
        "Day of Week": day_names[day_week_input]
    }

if "prediction" in st.session_state:
    st.sidebar.markdown(
        f"""
        <div class="prediction-card">
            Predicted Temperature:<br>
            {st.session_state["prediction"]:.2f} C
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div class="warning-card">
            No prediction yet. Click Predict Temperature.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# Main Dashboard Header
# =====================================================
st.markdown(
    '<div class="main-title">IoT Sensor Data Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">A machine learning dashboard for analyzing IoT sensor data and predicting temperature.</div>',
    unsafe_allow_html=True
)


# =====================================================
# Section 1: Dataset Summary
# =====================================================
st.header("1. Dataset Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Records", f"{df.shape[0]:,}")

with summary_col2:
    sensor_type = df["sensor_type"].iloc[0] if "sensor_type" in df.columns else "Unknown"
    st.metric("Sensor Type", sensor_type)

with summary_col3:
    temp_min = df["temperature"].min()
    temp_max = df["temperature"].max()
    st.metric("Temperature Range", f"{temp_min:.1f} - {temp_max:.1f} C")

with summary_col4:
    hum_min = df["humidity"].min()
    hum_max = df["humidity"].max()
    st.metric("Humidity Range", f"{hum_min:.0f} - {hum_max:.0f}")

st.markdown(
    """
    <div class="section-card">
    This dashboard uses IoT weather sensor readings. The main values used are timestamp, humidity, and temperature. 
    The timestamp was converted into time features such as hour, day, month, and day of week for machine learning prediction.
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Section 2: Sensor Data Trends
# =====================================================
st.header("2. Sensor Data Trends")

st.markdown(
    """
    <div class="section-card">
    These charts show how the sensor readings changed over time. They help explain the general behavior of the IoT sensor data.
    </div>
    """,
    unsafe_allow_html=True
)

trend_option = st.radio(
    "Select chart to display",
    ["Temperature Trend", "Humidity Trend", "Both"],
    horizontal=True
)

if trend_option in ["Temperature Trend", "Both"]:
    st.subheader("Temperature Trend Over Time")
    temp_chart = create_line_chart(
        df["timestamp"],
        df["temperature"],
        "Temperature Trend Over Time",
        "Timestamp",
        "Temperature (C)"
    )
    st.pyplot(temp_chart)

if trend_option in ["Humidity Trend", "Both"]:
    st.subheader("Humidity Trend Over Time")
    hum_chart = create_line_chart(
        df["timestamp"],
        df["humidity"],
        "Humidity Trend Over Time",
        "Timestamp",
        "Humidity"
    )
    st.pyplot(hum_chart)


# =====================================================
# Section 3: Model Performance Metrics
# =====================================================
st.header("3. Model Performance Metrics")

st.markdown(
    """
    <div class="section-card">
    The model used in this dashboard is Random Forest Regressor. It predicts temperature using humidity and time-based features.
    The model is evaluated using MAE, RMSE, and R2 Score.
    </div>
    """,
    unsafe_allow_html=True
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.metric("MAE", f"{mae:.3f}")

with metric_col2:
    st.metric("RMSE", f"{rmse:.3f}")

with metric_col3:
    st.metric("R2 Score", f"{r2:.3f}")

chart_col, table_col = st.columns([1.2, 1])

with chart_col:
    st.subheader("Actual vs Predicted Temperature")
    pred_chart = create_actual_vs_predicted_chart(y_test, y_pred)
    st.pyplot(pred_chart)

with table_col:
    st.subheader("Prediction Sample")
    st.dataframe(
        results.head(12),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# Section 4: Predicted Sensor Values
# =====================================================
st.header("4. Predicted Sensor Value")

st.markdown(
    """
    <div class="section-card">
    Use the controls on the left sidebar to enter humidity and time values. 
    After clicking the prediction button, the dashboard shows the predicted temperature.
    </div>
    """,
    unsafe_allow_html=True
)

if "prediction" in st.session_state:
    pred_col1, pred_col2 = st.columns([1, 1])

    with pred_col1:
        st.markdown(
            f"""
            <div class="prediction-card">
                Predicted Temperature: {st.session_state["prediction"]:.2f} C
            </div>
            """,
            unsafe_allow_html=True
        )

    with pred_col2:
        input_table = pd.DataFrame([st.session_state["inputs"]])
        st.dataframe(
            input_table,
            use_container_width=True,
            hide_index=True
        )
else:
    st.markdown(
        """
        <div class="warning-card">
        No prediction has been generated yet. Use the left sidebar and click Predict Temperature.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# Footer
# =====================================================
st.divider()
st.caption("IT351 - IoT Sensor Data Prediction Dashboard Using Machine Learning")
