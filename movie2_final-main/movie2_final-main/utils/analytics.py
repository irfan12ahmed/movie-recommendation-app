import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_language_insights(df):
    lang_counts = df['original_language'].value_counts().head(10)
    st.bar_chart(lang_counts)

def show_monthly_trends(df):
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['month'] = df['release_date'].dt.month
    monthly_counts = df['month'].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(monthly_counts.index, monthly_counts.values)
    ax.set_xticks(range(1,13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Releases")
    ax.set_title("Movie Release Trends by Month")
    st.pyplot(fig)