import streamlit as st

def dashboard_page(df):
    st.title("🎬 Movie Analytics Dashboard")
    st.markdown("Explore movie trends, genres, languages, and more!")

    st.subheader("1. Movies Released Per Year")
    df['year'] = df['release_date'].dt.year
    year_counts = df['year'].value_counts().sort_index()
    st.line_chart(year_counts)

    st.subheader("2. Movie Count by Language")
    language_counts = df['original_language'].value_counts().head(10)
    st.bar_chart(language_counts)

    st.subheader("3. Popular Genres")
    genre_counts = df['genres'].str.get_dummies(sep=", ").sum().sort_values(ascending=False).head(10)
    st.bar_chart(genre_counts)