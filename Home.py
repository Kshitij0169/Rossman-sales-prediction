import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import timedelta
from models.sarimax import train_arima
import plotly.graph_objects as go


# page headings
st.set_page_config(layout="wide", page_title="INFO7374: Algorithmic Digital Marketing")
st.markdown('<img src="https://brand.northeastern.edu/wp-content/uploads/2022/06/ac-grid-6-black.svg" alt="drawing" width="100"/>', unsafe_allow_html=True)
st.subheader("INFO7374: Final Project | Team 2: Adit Bhosale, Sowmya Chatti, Vasundhara Sharma")

with open("functionalities/home_page_content.html", mode="r",  encoding="utf8") as file:
    home_page_content = file.read()

st.markdown(home_page_content, unsafe_allow_html=True)