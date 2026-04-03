import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("customer_sales_data.csv")

# Sidebar Filters
product_filter = st.sidebar.selectbox("Select Product", df['Product Category'].unique())

# Filter Data
filtered_df = df[df['Product Category'] == product_filter]

# Show Sales Trend
st.title("Sales Dashboard")
st.write("### Sales Trend")
sales_trend = filtered_df.groupby("Purchase Date")["Total Purchase Amount"].sum()
st.line_chart(sales_trend)

# Show Top Customers
st.write("### Top Customers")
top_customers = filtered_df.groupby("Customer ID")["Total Purchase Amount"].sum().nlargest(5)
st.bar_chart(top_customers)

