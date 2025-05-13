import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def build_recommendation_model(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["content"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, df, cosine_sim):
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices].tolist()

def recommended_page(df):
    st.header("🎥 Recommended Movies")
    st.write("Get movie recommendations based on your favorite movie!")

    df["genres"] = df["genres"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["content"] = df["genres"] + " " + df["overview"]

    movie_titles = df["title"].dropna().unique().tolist()
    user_movie = st.selectbox("Start typing a movie title:", movie_titles)

    cosine_sim = build_recommendation_model(df)

    if st.button("Get Recommendations"):
        if user_movie.strip() == "":
            st.warning("Please select a movie title.")
        else:
            recommendations = get_recommendations(user_movie, df, cosine_sim)
            if recommendations:
                st.success(f"Here are some movies similar to '{user_movie}':")
                for rec in recommendations:
                    st.write(f"- {rec}")
            else:
                st.error("No recommendations found for the selected movie.")