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
# Page configuration
# =====================================================
st.set_page_config(
    page_title="IoT Sensor Data Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# Custom styling
# =====================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-align: left;
    }

    .subtitle {
        font-size: 1rem;
        color: inherit;
        opacity: 0.85;
        margin-bottom: 1.2rem;
        text-align: left;
    }

    .info-card {
        background: rgba(120, 120, 120, 0.12);
        border: 1px solid rgba(120, 120, 120, 0.28);
        border-radius: 14px;
        padding: 18px 20px;
        margin: 10px 0 22px 0;
        color: inherit;
        line-height: 1.7;
    }

    .small-card {
        background: rgba(120, 120, 120, 0.10);
        border: 1px solid rgba(120, 120, 120, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: inherit;
    }

    .prediction-box {
        background: rgba(40, 167, 69, 0.18);
        border: 1px solid rgba(40, 167, 69, 0.45);
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
        font-size: 1.1rem;
        font-weight: 700;
        color: inherit;
    }

    .warning-box {
        background: rgba(255, 193, 7, 0.16);
        border: 1px solid rgba(255, 193, 7, 0.45);
        border-radius: 12px;
        padding: 14px;
        margin-top: 12px;
        color: inherit;
    }

    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.10);
        border: 1px solid rgba(120, 120, 120, 0.20);
        padding: 14px;
        border-radius: 14px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(120, 120, 120, 0.25);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# Load and prepare data
# =====================================================
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


@st.cache_data
def preprocess_data(df):
    data = df.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp", "humidity", "temperature"])

    data["hour"] = data["timestamp"].dt.hour
    data["day_of_month"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    return data.sort_values("timestamp")


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

    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, X_test, y_test, y_pred, mae, rmse, r2, results, feature_importance


df_raw = load_data()
df = preprocess_data(df_raw)

model, X_test, y_test, y_pred, mae, rmse, r2, results, feature_importance = train_model(df)


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

feature_names = {
    "humidity": "Humidity",
    "hour": "Hour of Day",
    "day_of_month": "Day of Month",
    "month": "Month",
    "day_of_week": "Day of Week",
    "temperature": "Temperature"
}


# =====================================================
# Sidebar prediction controls
# =====================================================
st.sidebar.title("Prediction Controls")
st.sidebar.write("Enter sensor and time values, then click the button to predict temperature.")

humidity_value = st.sidebar.number_input(
    "Humidity",
    min_value=float(df["humidity"].min()),
    max_value=float(df["humidity"].max()),
    value=float(round(df["humidity"].mean(), 2)),
    step=0.5
)

hour_value = st.sidebar.selectbox(
    "Hour of Day",
    options=list(range(24)),
    index=12,
    format_func=lambda x: f"{x:02d}:00"
)

day_value = st.sidebar.number_input(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15,
    step=1
)

month_value = st.sidebar.selectbox(
    "Month",
    options=list(month_names.keys()),
    index=7,
    format_func=lambda x: month_names[x]
)

day_week_value = st.sidebar.selectbox(
    "Day of Week",
    options=list(day_names.keys()),
    index=0,
    format_func=lambda x: day_names[x]
)

predict_clicked = st.sidebar.button("Predict Temperature")

if predict_clicked:
    input_data = pd.DataFrame({
        "humidity": [humidity_value],
        "hour": [hour_value],
        "day_of_month": [day_value],
        "month": [month_value],
        "day_of_week": [day_week_value]
    })

    st.session_state["prediction"] = float(model.predict(input_data)[0])
    st.session_state["prediction_inputs"] = {
        "Humidity": humidity_value,
        "Hour of Day": f"{hour_value:02d}:00",
        "Day of Month": day_value,
        "Month": month_names[month_value],
        "Day of Week": day_names[day_week_value]
    }

if "prediction" in st.session_state:
    st.sidebar.markdown(
        f"""
        <div class="prediction-box">
            Predicted Temperature:<br>{st.session_state["prediction"]:.2f} C
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        """
        <div class="warning-box">
            No prediction yet. Click <b>Predict Temperature</b> to show the result.
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# Header
# =====================================================
st.markdown(
    '<div class="main-title">IoT Sensor Data Prediction Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An interactive dashboard for analyzing IoT weather sensor readings and predicting temperature using machine learning.</div>',
    unsafe_allow_html=True
)


# =====================================================
# KPI cards
# =====================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Records", f"{df.shape[0]:,}")

with col2:
    st.metric("Features Used", "5")

with col3:
    sensor_name = df["sensor_type"].iloc[0] if "sensor_type" in df.columns else "Unknown"
    st.metric("Sensor Type", sensor_name)

with col4:
    st.metric("Latest Temp", f"{df['temperature'].iloc[-1]:.2f} C")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("MAE", f"{mae:.3f}")

with m2:
    st.metric("RMSE", f"{rmse:.3f}")

with m3:
    st.metric("R2 Score", f"{r2:.3f}")

st.divider()


# =====================================================
# Helper chart functions
# =====================================================
def line_chart(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def scatter_chart(x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================
# Tabs
# =====================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dataset Overview",
    "Sensor Trends",
    "EDA Analysis",
    "ML Results",
    "Prediction"
])


# =====================================================
# Tab 1: Dataset Overview
# =====================================================
with tab1:
    st.header("Dataset Overview")

    st.markdown(
        """
        <div class="info-card">
        This dataset contains IoT readings from a weather sensor. The main columns used in this project are timestamp, humidity, and temperature. The timestamp column was converted into time-based features to help the machine learning model learn daily and monthly patterns.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1.15, 1])

    with c1:
        st.subheader("Sample Data")

        sample_data = df.head(15).copy()

        if "timestamp" in sample_data.columns:
            sample_data["timestamp"] = sample_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

        sample_data = sample_data.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "humidity": "Humidity",
            "temperature": "Temperature",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "sensor_type": "Sensor Type",
            "sensor_id": "Sensor ID",
            "hour": "Hour",
            "day_of_month": "Day of Month",
            "month": "Month",
            "day_of_week": "Day of Week"
        })

        number_cols = sample_data.select_dtypes(include=["float64", "float32"]).columns
        sample_data[number_cols] = sample_data[number_cols].round(2)

        st.dataframe(
            sample_data,
            use_container_width=True,
            hide_index=True
        )

    with c2:
        st.subheader("Basic Statistics")

        stats_table = df[[
            "humidity",
            "temperature",
            "hour",
            "day_of_month",
            "month",
            "day_of_week"
        ]].describe().round(2)

        stats_table = stats_table.rename(columns=feature_names)
        stats_table.index = stats_table.index.map({
            "count": "Count",
            "mean": "Mean",
            "std": "Std",
            "min": "Min",
            "25%": "25%",
            "50%": "50%",
            "75%": "75%",
            "max": "Max"
        })

        st.dataframe(
            stats_table,
            use_container_width=True
        )

    st.subheader("Missing Values")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]
    missing["Column"] = missing["Column"].replace({
        "id": "ID",
        "timestamp": "Timestamp",
        "humidity": "Humidity",
        "temperature": "Temperature",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "sensor_type": "Sensor Type",
        "sensor_id": "Sensor ID",
        "hour": "Hour",
        "day_of_month": "Day of Month",
        "month": "Month",
        "day_of_week": "Day of Week"
    })

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# Tab 2: Sensor Trends
# =====================================================
with tab2:
    st.header("Sensor Data Trends")

    st.markdown(
        """
        <div class="info-card">
        These charts show how temperature and humidity changed over time. This helps us understand the behavior of the IoT sensor readings before applying machine learning.
        </div>
        """,
        unsafe_allow_html=True
    )

    chart_choice = st.radio(
        "Choose trend chart",
        ["Temperature", "Humidity", "Both"],
        horizontal=True
    )

    if chart_choice in ["Temperature", "Both"]:
        st.subheader("Temperature Trend Over Time")
        fig_temp = line_chart(
            df["timestamp"],
            df["temperature"],
            "Temperature Trend Over Time",
            "Timestamp",
            "Temperature (C)"
        )
        st.pyplot(fig_temp)

    if chart_choice in ["Humidity", "Both"]:
        st.subheader("Humidity Trend Over Time")
        fig_hum = line_chart(
            df["timestamp"],
            df["humidity"],
            "Humidity Trend Over Time",
            "Timestamp",
            "Humidity"
        )
        st.pyplot(fig_hum)


# =====================================================
# Tab 3: EDA Analysis
# =====================================================
with tab3:
    st.header("Exploratory Data Analysis")

    st.markdown(
        """
        <div class="info-card">
        EDA is used to find patterns in the dataset. The scatter plot and correlation matrix show that humidity and time features are useful for predicting temperature.
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Humidity vs Temperature")
        fig_scatter = scatter_chart(
            df["humidity"],
            df["temperature"],
            "Relationship Between Humidity and Temperature",
            "Humidity",
            "Temperature (C)"
        )
        st.pyplot(fig_scatter)

    with c2:
        st.subheader("Correlation Matrix")

        corr_columns = ["humidity", "temperature", "hour", "day_of_month", "month", "day_of_week"]
        corr = df[corr_columns].corr()
        corr_display_names = [feature_names.get(col, col) for col in corr_columns]

        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        image = ax_corr.imshow(corr, aspect="auto")
        ax_corr.set_xticks(range(len(corr_columns)))
        ax_corr.set_yticks(range(len(corr_columns)))
        ax_corr.set_xticklabels(corr_display_names, rotation=40, ha="right")
        ax_corr.set_yticklabels(corr_display_names)

        for i in range(len(corr_columns)):
            for j in range(len(corr_columns)):
                ax_corr.text(
                    j,
                    i,
                    f"{corr.iloc[i, j]:.2f}",
                    ha="center",
                    va="center"
                )

        ax_corr.set_title("Correlation Between Features")
        fig_corr.colorbar(image)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)

    st.subheader("Feature Importance")

    feature_importance_clean = feature_importance.copy()
    feature_importance_clean["Feature"] = feature_importance_clean["Feature"].replace(feature_names)
    feature_importance_clean["Importance"] = feature_importance_clean["Importance"].round(4)

    st.dataframe(
        feature_importance_clean,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# Tab 4: ML Results
# =====================================================
with tab4:
    st.header("Machine Learning Results")

    st.markdown(
        """
        <div class="info-card">
        A Random Forest Regressor was trained to predict temperature using humidity and time-based features. The model was evaluated using MAE, RMSE, and R2 Score.
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.metric("Mean Absolute Error", f"{mae:.3f}")

    with b:
        st.metric("Root Mean Squared Error", f"{rmse:.3f}")

    with c:
        st.metric("R2 Score", f"{r2:.3f}")

    st.subheader("Actual vs Predicted Temperature")

    fig_pred, ax_pred = plt.subplots(figsize=(8, 5))
    ax_pred.scatter(y_test, y_pred, alpha=0.75)

    min_val = min(float(y_test.min()), float(y_pred.min()))
    max_val = max(float(y_test.max()), float(y_pred.max()))

    ax_pred.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2
    )

    ax_pred.set_title("Actual vs Predicted Temperature")
    ax_pred.set_xlabel("Actual Temperature (C)")
    ax_pred.set_ylabel("Predicted Temperature (C)")
    ax_pred.grid(True, alpha=0.3)
    fig_pred.tight_layout()
    st.pyplot(fig_pred)

    st.subheader("Prediction Results Sample")

    results_clean = results.head(25).copy()
    results_clean["Actual Temperature"] = results_clean["Actual Temperature"].round(2)
    results_clean["Predicted Temperature"] = results_clean["Predicted Temperature"].round(2)

    st.dataframe(
        results_clean,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# Tab 5: Prediction
# =====================================================
with tab5:
    st.header("Temperature Prediction")

    st.markdown(
        """
        <div class="info-card">
        Use the controls in the left sidebar to enter humidity and time values. Then click the prediction button to generate a new temperature prediction.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "prediction" in st.session_state:
        st.subheader("Latest Prediction")

        p1, p2 = st.columns([1, 1])

        with p1:
            st.markdown(
                f"""
                <div class="prediction-box">
                    Predicted Temperature: {st.session_state["prediction"]:.2f} C
                </div>
                """,
                unsafe_allow_html=True
            )

        with p2:
            st.markdown(
                "<div class='small-card'><b>Input Values Used</b></div>",
                unsafe_allow_html=True
            )

            input_table = pd.DataFrame([st.session_state["prediction_inputs"]])

            st.dataframe(
                input_table,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.markdown(
            """
            <div class="warning-box">
            No prediction has been generated yet. Use the sidebar and click Predict Temperature.
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()
st.caption("IT351 - IoT Sensor Data Prediction Dashboard using Machine Learning")
