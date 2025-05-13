import streamlit as st
import pandas as pd
from home import home_page
from dashboard import dashboard_page
from release_forecast import release_forecast_page
from recommended import recommended_page
from popularity_predictor import popularity_predictor_page
from revenue_predictor import revenue_predictor_page
from auth_pages import login_page, signup_page, logout
from auth import init_db
import os

# Set page configuration
st.set_page_config(
    page_title="Movie Analytics & Recommendation App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the database
try:
    init_db()
except Exception as e:
    st.error(f"❌ Database initialization failed: {e}")

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"

# Load data with caching
@st.cache_data
def load_data():
    data_path = os.path.join("data", "movie_.csv")

    if not os.path.exists(data_path):
        st.error(f"❌ Error: The file {data_path} was not found.")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    return df.dropna(subset=['release_date'])

# Load the dataset
df = load_data()

# Main app logic
def main_app():
    st.sidebar.title("📊 Movie App")
    st.sidebar.write(f"Welcome, **{st.session_state.username}**!")

    if st.sidebar.button("Logout"):
        logout()

    page = st.sidebar.radio(
        "Navigate to",
        [
            "Home",
            "Dashboard",
            "Release Pattern Forecast",
            "Recommended",
            "Popularity Predictor",
            "Revenue Predictor"
        ]
    )

    if page == "Home":
        home_page()
    elif page == "Dashboard":
        dashboard_page(df)
    elif page == "Release Pattern Forecast":
        release_forecast_page(df)
    elif page == "Recommended":
        recommended_page(df)
    elif page == "Popularity Predictor":
        popularity_predictor_page(df)
    elif page == "Revenue Predictor":
        revenue_predictor_page(df)

# Route to appropriate page
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
else:
    main_app()
