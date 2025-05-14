import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def release_forecast_page(df):
    st.header("🔮 Release Forecast")
    df['year'] = df['release_date'].dt.year
    yearly_releases = df['year'].value_counts().sort_index()
    
    if len(yearly_releases) >= 5:
        arima = ARIMA(yearly_releases, order=(1,1,1))
        model = arima.fit()
        forecast = model.forecast(steps=5)
        st.line_chart(pd.concat([yearly_releases, forecast]))
    else:
        st.warning("Not enough data to forecast.")