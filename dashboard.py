import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# --- Load Data ---
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "customer_sales_data.csv")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
else:
    st.error("Dataset not found!")
    st.stop()

# --- Sidebar ---
st.sidebar.header("Filters")
product_filter = st.sidebar.selectbox("Select Product Category", df['Product Category'].unique())

# --- Filtered Data ---
filtered_df = df[df['Product Category'] == product_filter]

# --- Main Dashboard ---
st.title("🚀 Sales Analysis & Forecasting")

# 1. Sales Trend
st.subheader(f"Monthly Sales Trend: {product_filter}")
sales_trend = filtered_df.groupby(df['Purchase Date'].dt.date)["Total Purchase Amount"].sum()
st.line_chart(sales_trend)

# 2. Sales Prediction (Machine Learning)
st.divider()
st.subheader("🔮 Future Sales Forecast")

# Prepare ML Data for the specific category
df_ml = filtered_df.copy()
df_ml['Month'] = df_ml['Purchase Date'].dt.month
df_ml['Year'] = df_ml['Purchase Date'].dt.year
ml_data = df_ml.groupby(['Year', 'Month'])['Total Purchase Amount'].sum().reset_index()

if len(ml_data) > 1:
    X = ml_data[['Year', 'Month']]
    y = ml_data['Total Purchase Amount']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for the next month
    last_month = ml_data['Month'].iloc[-1]
    last_year = ml_data['Year'].iloc[-1]
    
    next_month = 1 if last_month == 12 else last_month + 1
    next_year = last_year + 1 if last_month == 12 else last_year
    
    prediction = model.predict([[next_year, next_month]])
    
    col1, col2 = st.columns(2)
    col1.metric("Target Month", f"{next_month}/{next_year}")
    col2.metric("Predicted Revenue", f"${prediction[0]:,.2f}")
else:
    st.info("Not enough historical data to generate a forecast for this category.")

# 3. Top Customers
st.divider()
st.subheader("🏆 Top 5 Customers")
top_customers = filtered_df.groupby("Customer ID")["Total Purchase Amount"].sum().nlargest(5)
st.bar_chart(top_customers)