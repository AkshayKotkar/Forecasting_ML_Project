#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
import base64

# Set Streamlit app title and page icon
st.set_page_config(
    page_title='Sales Forecasting ',
    page_icon=':chart_with_upwards_trend:'
)

# Load your dataset
@st.cache_resource
def load_data():
    try:
        data = pd.read_csv(r"C:\Users\aksha\Desktop\Internship\Ai Varient\Demand Forecasting\train (2).csv")  # Update with your dataset path
        return data
    except FileNotFoundError:
        st.error("Dataset file not found. Please check the file path.")
        st.stop()

data = load_data()

st.title('Time Series Forecasting')

# Load the pre-trained Sarimax model from a .pkl file
@st.cache_resource
def load_model():
    try:
        with open(r"C:\Users\aksha\Desktop\Internship\Ai Varient\Demand Forecasting\Deploy.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error("model file not found. Please check the file path.")
        st.stop()

model = load_model()

# Forecast future sales
st.sidebar.header('Forecast Day')

future_periods = st.sidebar.number_input('Enter the number of periods to forecast:', min_value=1, max_value=104, value=7)

# Forecast future sales with Sarimax
forecast = model.forecast(steps=future_periods)

# Assuming 'data' is your DataFrame containing the sales data
last_date = pd.to_datetime(data["date"].iloc[-1])
forecast_interval = 7
future_dates = [last_date + relativedelta(days=x) for x in range(1, forecast_interval * future_periods+1, forecast_interval)]

forecast_df = pd.DataFrame({'date': future_dates, 'sales_forecast': forecast})
forecast_df.set_index('date', inplace=True)

# Customize forecast plot
st.sidebar.header('Forecast Plot')

# User can select plot type
plot_type = st.sidebar.selectbox('Select Plot Type', ['Line Plot', 'Bar Chart'], index=0)

# User can select plot color
plot_color = st.sidebar.color_picker('Select Plot Color', '#1f77b4')

# Function to create the forecast plot
def create_forecast_plot(forecast_data, plot_type, plot_color):
    fig = None
    if plot_type == 'Line Plot':
        fig = px.line(forecast_data, x=forecast_data.index, y='sales_forecast', labels={'sales_forecast': 'Sales'})
    elif plot_type == 'Bar Chart':
        fig = px.bar(forecast_data, x=forecast_data.index, y='sales_forecast', labels={'sales_forecast': 'Sales'})
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Sales')
    fig.update_traces(marker_color=plot_color)
    return fig

# Create the customized forecast plot
st.write("### Forecast Plot")
customized_plot = create_forecast_plot(forecast_df, plot_type, plot_color)
st.plotly_chart(customized_plot)

# Show the forecasted sales values as a table
st.write("### Forecasted Sales Values")
st.dataframe(forecast_df)

# Add a button to download the forecasted values as a CSV file
if st.button("Download Forecasted Values as CSV"):
    csv = forecast_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert the CSV data to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_values.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Load historical sales data
historical_data = data[['date', 'sales']]
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data.set_index('date', inplace=True)

# Resample historical data to 7-day intervals and calculate sum
historical_data_resampled = historical_data.resample('7D').sum()/100

# Checkbox to show/hide historical data
show_historical_data = st.checkbox('Historical Data', value=False)

if show_historical_data:
    st.subheader('Historical Data')
    st.dataframe(historical_data_resampled)