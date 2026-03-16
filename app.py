import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="Smart Traffic Prediction", layout="wide")

st.title("🚦 Smart Traffic Congestion Prediction Dashboard")

# Load Dataset
df = pd.read_csv("traffic.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature Engineering
df["DateTime"] = pd.to_datetime(df["DateTime"])

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

df = df.drop(["ID", "DateTime"], axis=1)

# Sidebar
st.sidebar.header("Navigation")

page = st.sidebar.selectbox(
    "Select Page",
    ["EDA Analysis", "Model Training", "Prediction"]
)

# ================= EDA PAGE =================

if page == "EDA Analysis":

    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vehicle Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Vehicles"], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Traffic by Hour")
        fig, ax = plt.subplots()
        sns.lineplot(x="Hour", y="Vehicles", data=df, ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Traffic per Junction")
        fig, ax = plt.subplots()
        sns.boxplot(x="Junction", y="Vehicles", data=df, ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# ================= MODEL PAGE =================

elif page == "Model Training":

    st.header("Model Training & Comparison")

    X = df.drop("Vehicles", axis=1)
    y = df["Vehicles"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    dt = DecisionTreeRegressor()
    rf = RandomForestRegressor()
    gb = GradientBoostingRegressor()

    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    dt_pred = dt.predict(X_test)
    rf_pred = rf.predict(X_test)
    gb_pred = gb.predict(X_test)

    scores = [
        r2_score(y_test, lr_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, gb_pred),
    ]

    models = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ]

    results = pd.DataFrame({
        "Model": models,
        "Accuracy": scores
    })

    st.subheader("Model Accuracy")
    st.dataframe(results)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=results, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest)")

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    importance.sort_values().plot(kind="barh", ax=ax)
    st.pyplot(fig)

# ================= PREDICTION PAGE =================

else:

    st.header("Traffic Prediction")

    year = st.slider("Year", 2015, 2030, 2024)
    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 12)
    dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 3)
    junction = st.slider("Junction", 1, 4, 1)

    input_data = pd.DataFrame({
        "Junction": [junction],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Hour": [hour],
        "DayOfWeek": [dayofweek]
    })

    X = df.drop("Vehicles", axis=1)
    y = df["Vehicles"]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    prediction = rf.predict(input_data)

    st.success(f"🚗 Predicted Traffic Vehicles: {int(prediction[0])}")