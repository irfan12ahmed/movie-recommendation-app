import streamlit as st
from auth import add_user, authenticate_user

# Login Page
def login_page():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
            # Redirect to the Home page
            st.session_state.page = "Home"
        else:
            st.error("Invalid username or password.")

    # Add a link to navigate to the signup page
    st.write("Don't have an account?")
    if st.button("Sign Up"):
        st.session_state.page = "signup"

# Signup Page
def signup_page():
    st.title("📝 Sign Up")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        try:
            add_user(username, password)
            st.success("User registered successfully! Please log in.")
            st.session_state.page = "login"
        except Exception as e:
            st.error("Username already exists. Please choose a different one.")

# Logout Functionality
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "login"