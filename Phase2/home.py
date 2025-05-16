import streamlit as st

# Page config
st.set_page_config(
    page_title="DEPI Diabetes Detection",
    layout="centered",
    page_icon="🩺"
)

# Background image using CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom right, #98FF98, #ffffff);
        background-attachment: fixed;
    }
    </style>

    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #2C3E50;'>🩺 DEPI Diabetes Detection</h1>
    <p style='text-align: center; font-size:18px; color: #34495E;'>
        Welcome to the DEPI system – your gateway to understanding and predicting diabetes.<br>
        Navigate through the tools below to either test yourself or explore detailed data analysis.
    </p>
    """,
    unsafe_allow_html=True
)

# Buttons
c1, c2 = st.columns(2)

with c1:
    st.link_button("🧪 Test Yourself", url="/input_form", use_container_width=True)

with c2:
    st.link_button("📊 People with Diabetes", url="/analysis", use_container_width=True)
