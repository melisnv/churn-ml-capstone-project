import streamlit as st
from predict_page import show_predict_page
from analyze_page import show_analysis
from xgboost_page import show_xgboost_page
from segmentation_page import  show_segmentation_page
from cluster_page import show_cluster_page

page = st.sidebar.selectbox("Pages",("Predict","Analysis","XGBoost Model","Segmentation","Cluster"))

if page == "Predict":
    show_predict_page()
elif page == "XGBoost Model":
    show_xgboost_page()
elif page == "Segmentation":
    show_segmentation_page()
elif page == "Cluster":
    show_cluster_page()
else:
    show_analysis()