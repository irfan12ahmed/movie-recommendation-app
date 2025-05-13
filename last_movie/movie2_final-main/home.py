import streamlit as st
import os
from PIL import Image

def home_page():
    st.title("🍿 Movie Recommendation Dashboard")
    st.subheader("✨ Recent Movie Picks")

    # Set up 3 columns for the first row
    cols = st.columns(3)

    # Movie 1
    with cols[0]:
        title = "Movie Title 1"
        release_year = "Year 1"
        image_file = "1.webp"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")

    # Movie 2
    with cols[1]:
        title = "Movie Title 2"
        release_year = "Year 2"
        image_file = "2.webp"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")

    # Movie 3
    with cols[2]:
        title = "Movie Title 3"
        release_year = "Year 3"
        image_file = "3.jpg"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")

    # Set up 3 columns for the second row
    cols = st.columns(3)

    # Movie 4
    with cols[0]:
        title = "Movie Title 4"
        release_year = "Year 4"
        image_file = "4.jpeg"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")

    # Movie 5
    with cols[1]:
        title = "Movie Title 5"
        release_year = "Year 5"
        image_file = "5.jpg"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")

    # Movie 6
    with cols[2]:
        title = "Movie Title 6"
        release_year = "Year 6"
        image_file = "6.jpg"
        image_path = os.path.join("posters", image_file)
        if not os.path.exists(image_path):
            st.warning(f"⚠️ Poster file not found: {image_file}")
            image_path = os.path.join("posters", "default.jpg")
        st.image(Image.open(image_path), use_column_width=True)
        st.caption(f"🎬 {title} ({release_year})")