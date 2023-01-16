import streamlit as st
from predict_page import show_predict_page
from analyze_page import show_analysis


page = st.sidebar.selectbox("Analysis or Predict",("Predict","Analysis"))


if page == "Predict":
    show_predict_page()
else:
    show_analysis()