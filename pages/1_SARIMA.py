import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import timedelta
from models.sarimax import train_arima
import plotly.graph_objects as go


st.set_page_config(layout="wide", page_title="SARIMA")
st.header("SARIMA: Seasonal Autoregressive Integrated Moving Average Exogenous model")


# initializing variables
store_type, store_id, flag_weekly_monthly = None, None, None
button_view_sales = None
button_predict_sales = None

with open("data/store_mappings.json", 'r') as file:
    store_dict = json.load(file)

try:
    col_store_type, col_store_id, col_radio, col_date = st.columns((1, 1, 1, 1))

    with col_store_type:
        store_type = st.selectbox(label='Select the Store Type', placeholder="Store Type", index=None, options=store_dict.keys(), key=2001)
    with col_store_id:
        if store_type:
            store_id = st.selectbox(label='Select the Store ID', placeholder="Store ID", index=None, options=store_dict[store_type], key=2002)
    with col_radio:
        if store_id:
            df = pd.read_csv("data/train.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values(by='Date', inplace=True)
            df = df[df['Store'] == store_id]
            df = df[['Date', 'Sales']]
            flag_weekly_monthly = st.selectbox("See weekly or monthly sales", options=['Weekly', 'Monthly', 'Raw'], index=None, key=2003)

    with col_date:
        if flag_weekly_monthly:
            df_indexed = df.set_index('Date', drop=False)
            if flag_weekly_monthly == "Weekly":
                df_indexed = df_indexed.resample('W').mean(numeric_only=True)
            elif flag_weekly_monthly == "Monthly":
                df_indexed = df_indexed.resample('M').mean(numeric_only=True)
            date_end_prediction = st.date_input("Pick End Prediction Date:",
                                                value=None,
                                                min_value=df['Date'].min().date() - timedelta(days=60),
                                                max_value=df['Date'].max().date() + timedelta(days=365))
            if date_end_prediction:
                col_space_1, col_space_2 = st.columns(2)
                with col_space_2:
                    button_view_sales = st.button("View Sales", key=1001, use_container_width=True)

        # display past sales data
    if button_view_sales:
        fig = px.line(df_indexed, x=df_indexed.index.values, y="Sales", markers=True)
        fig.update_layout(
            title=f'{flag_weekly_monthly} Sales of Store Type: {store_type.upper()} | Store ID: {store_id}',
            xaxis_title='Date',
            yaxis_title='Sales in USD',
        )
        fig.update_traces(marker_color='rgb(158,202,225)', line=dict(color='#0096c7', width=3), marker=dict(color="#0077b6"))
        st.plotly_chart(fig, use_container_width=True)

        pred, pred_ci = train_arima(df=df_indexed, end_date=date_end_prediction)

        test = pd.concat([df_indexed, pred.predicted_mean])
        test = test.rename(columns={0: "PredicatedSales"})
        fig = px.line(test, x=test.index.values, y="Sales")
        #fig.add_scatter(x=test.index.values, y=test["PredicatedSales"])
        fig.update_traces(marker_color='rgb(158,202,225)', line=dict(color='#0096c7', width=3))

        fig.add_trace(go.Scatter(
            x=pred_ci.index,
            y=pred_ci.iloc[:, 0],
            line=dict(width=0.2),
            line_color='#fde4cf',
            name='',
        ))

        fig.add_trace(go.Scatter(
            x=pred_ci.index,
            y=pred_ci.iloc[:, 1],
            line=dict(width=0.2),
            line_color='#fde4cf',
            fill='tonexty',
            name='Predicted Range',
        ))

        fig.add_trace(go.Scatter(x=test.index.values,
                               y=test["PredicatedSales"],
                               name="Predicted Sales",
                               line=dict(width=3),
                               marker=dict(color="#f72585")))

        fig.update_layout(
            title_font_size=24,
            title=f'Predicted Sales of Store Type: {store_type.upper()} | Store ID: {store_id}',
            xaxis_title='Date',
            yaxis_title='Sales in USD'
        )
        fig.update_xaxes(showgrid=True)
        #fig.update_traces(marker_color='rgb(158,202,225)', line=dict(color='orange', width=4))
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"OOPS:heavy_exclamation_mark: An error occurred. Please restart the application :new_moon_with_face:\nError: {e}")