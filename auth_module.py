# 🔐 Supabase Auth Integration Module
# File: auth_module.py

import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def login_form():
    st.subheader("🔐 Login to Continue")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        sb = init_supabase()
        try:
            user = sb.auth.sign_in_with_password({"email": email, "password": password})
            st.session_state["user"] = user
            st.success("Logged in successfully")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

def signup_form():
    st.subheader("🆕 Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")
    if st.button("Create Account"):
        sb = init_supabase()
        try:
            sb.auth.sign_up({"email": email, "password": password})
            st.success("Account created! Check your email for confirmation.")
        except Exception as e:
            st.error(f"Signup failed: {e}")

def logout():
    if st.button("🚪 Logout"):
        st.session_state.pop("user", None)
        st.success("Logged out")
        st.rerun()

def auth_gate():
    if "user" not in st.session_state:
        tabs = st.tabs(["Login", "Sign Up"])
        with tabs[0]:
            login_form()
        with tabs[1]:
            signup_form()
        st.stop()
    else:
        user_email = st.session_state['user'].user.email
        short_email = user_email[:6] + "..."
        with st.sidebar:
            st.markdown(f"👤 Logged in as - `{short_email}`")
            logout()
