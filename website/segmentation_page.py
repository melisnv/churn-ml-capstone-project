import streamlit as st
import pandas as pd
import plotly.express as ex
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def read_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop([
                         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                         'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
                     axis='columns')
    return data


def create_afm(dataframe, csv=False):
    afm = pd.DataFrame()
    afm["client_id"] = dataframe["CLIENTNUM"]
    afm["activity"] = (12 - dataframe["Months_Inactive_12_mon"])
    afm["frequency"] = dataframe["Contacts_Count_12_mon"]
    afm["monetary"] = dataframe["Total_Trans_Amt"]

    # activity score
    afm["activity_score"] = afm.apply(lambda _: ' ', axis=1)
    afm.loc[afm["activity"] == 12, "activity_score"] = 5
    afm.loc[afm["activity"] == 11, "activity_score"] = 4
    afm.loc[afm["activity"] == 10, "activity_score"] = 4
    afm.loc[afm["activity"] == 9, "activity_score"] = 3
    afm.loc[afm["activity"] == 8, "activity_score"] = 3
    afm.loc[afm["activity"] == 7, "activity_score"] = 2
    afm.loc[afm["activity"] == 6, "activity_score"] = 1

    # monetary score
    afm["monetary_score"] = pd.qcut(afm["monetary"], 5, labels=[1, 2, 3, 4, 5])

    # frequency score
    afm["frequency_score"] = pd.qcut(afm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    afm["AFM_score"] = (afm["frequency_score"].astype(str) + afm["activity_score"].astype(str))

    # AFM naming
    seg = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'regular customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions',
    }

    afm["segment"] = afm["AFM_score"].replace(seg, regex=True)
    if csv:
        afm.to_csv("afm.csv")
    return afm


def show_segmentation_page():
    df = read_data("./data/BankChurners.csv")

    st.title("Customer Segmentation with AFM")
    st.title("")
    st.markdown("After we came to the conclusion that our data is not suitable for the RFM model, we decided to "
                "create an approach inspired by the RFM model. For this, we first investigated whether we could "
                "create a personal model.")

    st.markdown("We came to the conclusion that it is possible to create personalized segmentation rules for customer "
                "segmentation instead of using the recency, frequency, and monetary (RFM) model. There are many "
                "different ways to segment the customers, and the approach chosen will depend on the specific "
                "business goals and the characteristics of the customer base.")
    st.dataframe(df.tail(2))

    st.markdown("Bank customer segmentation is the process of dividing bank customers into groups based on shared "
                "characteristics or behaviors. This can be done for a variety of purposes, such as targeting "
                "marketing efforts, improving customer service, or identifying potential cross-selling opportunities.")

    code_activity1 = '''
        afm["activity"] = (12 - dataframe["Months_Inactive_12_mon"])
        '''
    st.code(code_activity1, language="python")

    code_activity2 = '''
    afm.loc[afm["activity"] == 12, "activity_score"] = 5
    afm.loc[afm["activity"] == 11, "activity_score"] = 4
    afm.loc[afm["activity"] == 10, "activity_score"] = 4
    afm.loc[afm["activity"] == 9, "activity_score"] = 3
    afm.loc[afm["activity"] == 8, "activity_score"] = 3
    afm.loc[afm["activity"] == 7, "activity_score"] = 2
    afm.loc[afm["activity"] == 6, "activity_score"] = 1
    '''
    st.code(code_activity2, language="python")

    st.markdown("In our model, we made customer classification by using the 'Activity' measure as a method. Our "
                "underlying motivation is that the customer's activity information in the last year, included in the "
                "data, is an important representation of whether or not he will leave the credit card service.")


    afm = create_afm(df, True)
    st.dataframe(afm[8:10])

    st.title("Propotion of Client Levels")
    img_cust_dist = Image.open('website/visuals/customer_distribution.png')
    st.image(img_cust_dist, caption='Client Segments')

    st.title("Distribution of Client Activity")
    img_activity = Image.open('website/visuals/user_activity.png')
    st.image(img_activity, caption='Client Activity')

    st.title("Distribution of Client Transaction")
    img_transac = Image.open('website/visuals/customer_transaction.png')
    st.image(img_transac, caption='Client Transaction')
