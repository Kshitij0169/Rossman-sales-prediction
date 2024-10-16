import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt

st.title('Predict Store Sales')

# User Inputs
store_id = st.number_input('Select Store ID: ', min_value=1, max_value=1115, value=3, key='store_id')
date_input = st.date_input('Select Date:', pd.to_datetime('today'))
duration = st.number_input('Select number of days you want to forecast sales for:', min_value=15, max_value=60, value=15)
promo_input = st.radio('Are you planning on offering any promotions?', ['Yes', 'No'], key='promo_input')
if promo_input == 'Yes':
    promo_dates = st.multiselect('Select the dates you want to offer promotions:', pd.date_range(start=date_input, periods=duration, freq='D'))
holiday_input = st.radio('Are there any holidays in this period?', ['Yes', 'No'], key='holiday_input')
if holiday_input == 'Yes':
    holiday_dates = st.multiselect('Select dates for holidays:', pd.date_range(start=date_input, periods=duration, freq='D'))



if st.button('Submit'):

    if promo_input == 'Yes':
        promo_input = 1
    else:
        promo_input = 0

    if holiday_input == 'Yes':
        holiday_input = 1
    else:
        holiday_input = 0


    # Based on user selected date and range, creating a date range series
    date_range = pd.date_range(start=date_input, periods=duration, freq='D')
    
    # Creating a temporary dataframe based on user inputs 
    t_df = pd.DataFrame({'ds': date_range, 'SchoolHoliday': holiday_input, 'Promo': promo_input})

    # Update SchoolHoliday and Promo columns based on user-selected dates, 0 for dates not selected and 1 for the dates selected
    if holiday_input == 1:
        t_df.loc[~t_df['ds'].isin(holiday_dates), 'SchoolHoliday'] = 0
    if promo_input == 1:
     t_df.loc[~t_df['ds'].isin(promo_dates), 'Promo'] = 0

    
    # Dataframe for training the model
    df_train = pd.read_csv('data/train.csv')

    # Filtering Open stores since closed stores have 0 Sales
    df_train = df_train[df_train['Open']==1]
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train['Sales'] = pd.to_numeric(df_train['Sales'])
    df_train = df_train[df_train.Store==store_id]
    prophet_df = df_train[['Date', 'Sales', 'SchoolHoliday','Promo']]
    prophet_df.rename(columns={'Date': 'ds', 'Sales': 'y'}, inplace=True)
    prophet_df_train = prophet_df[['ds', 'y', 'SchoolHoliday','Promo']]

    # Prophet model
    model = Prophet()
    # Adding Regressors 
    model.add_regressor('Promo',standardize= 'False')
    model.add_regressor('SchoolHoliday',standardize= 'False')
    model.fit(prophet_df_train)


    
    pred = model.predict(t_df)
    
    
    
   # Create Altair chart
    chart = alt.Chart(pred).mark_line().encode(
        x='ds', 
        y='yhat').properties(width=600, height=400)

    # Add labels to axes
    chart = chart.properties(title='Sales Prediction')
    chart = chart.configure_title(fontSize=16)
    chart = chart.configure_axisX(title='Date')
    chart = chart.configure_axisY(title='Sales Prediction')

    # Display the Altair chart
    st.markdown(f"Here's a line chart showing sales predictions for the store with id {store_id} for the coming {duration} days starting from {date_input}:")

    st.altair_chart(chart, use_container_width=True)

   