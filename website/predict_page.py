import pandas as pd
import streamlit as st
import pickle
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import RobustScaler


def load_model():
    with open('../website/ml_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data["model"]

        # Save the model in the older version
        import xgboost as xgb
        model.save_model('old_model.model')

        # Load the model in the current version
        loaded_model = xgb.Booster()
        loaded_model.load_model('old_model.model')

        return loaded_model


xgb_model = load_model()


def show_predict_page():
    st.title("Client Churn Prediction")
    st.write("""## Fill the information to predict whether client will leave""")

    gender = (
        'Female',
        'Male'
    )

    education = (
        'Uneducated',
        'Highschool',
        'College',
        'Graduate',
        'Post-graduate',
        'Doctorate'
    )

    income = (
        'Less than $40K',
        '$40K - $60K',
        '$60K - $80K',
        '$80K - $120K',
        '$120K +'
    )

    status = (
        'Single',
        'Married',
        'Divorced'
    )

    gen = st.selectbox("Gender",  {"Female": 0, "Male": 1})

    age = st.slider("Age", 18, 100, 25)
    age_dict = {(25, 35): 0, (35, 45): 1, (45, 55): 2, (55, 65): 3, (65, float('inf')): 4}
    for age_range, age_group in age_dict.items():
        if age_range[0] <= age <= age_range[1]:
            age = age_group
            break

    edu = st.selectbox("Education Level", education)
    edu_map = {'Uneducated': 0, 'High School': 0, 'Graduate': 1, 'College': 1, 'Post-Graduate': 1, 'Doctorate': 2}
    edu = edu_map[edu]

    inactive = st.slider("Inactivity", 0, 100, 25)

    stat = st.selectbox("Status", status)
    stat_map = {'Single': 0, 'Married': 1, 'Divorced': 0}
    stat = stat_map[stat]

    client_history = st.slider("Months as Client", 13, 60, 2)
    history_dict = {(0, 25): 0, (25, 35): 1, (35, 45): 2, (45, 60): 3, (60, float('inf')): 4}
    for hist_range, hist_group in history_dict.items():
        if hist_range[0] <= client_history <= hist_range[1]:
            client_history = hist_group
            break

    inc = st.selectbox("Income", {'Less than $40K': 1, '$40K - $60K': 2, '$80K - $120K': 4, '$60K - $80K': 3, '$120K +': 5})
    transaction = st.slider("Total Transaction", 0, int(round(18484.000000)), int(round(510.000000)))

    feature_ranges = {
        "Customer_Age": age,
        "Gender": gen,
        "Dependent_count": (1.0, 3),
        "Education_Level": edu,
        "Marital_Status": stat,
        "Income_Category": inc,
        "Card_Category": (0.0, 3.0),
        "Months_on_book": client_history,
        "Total_Relationship_Count": (1, 4),
        "Months_Inactive_12_mon": inactive,
        "Contacts_Count_12_mon": (2.0, 4.0),
        "Credit_Limit": (0.3654273127753303, 3.520352422907489),
        "Total_Revolving_Bal": (0.895438596491228, 0.8708771929824561),
        "Avg_Open_To_Buy": (0.4067022086824067, 3.6372370964907135),
        "Total_Amt_Chng_Q4_Q1": (3.2280701754385968, 11.671052631578949),
        "Total_Trans_Amt": transaction,
        "Total_Trans_Ct": (1.5833333333333333, 2.0),
        "Total_Ct_Chng_Q4_Q1": (2.9745762711864407, 12.76271186440678),
        "Avg_Utilization_Ratio": (0.3666666666666666, 1.7145833333333331),
        "Transaction_Count_per_Contact": (1.2777777777777777, 3.87037037037037),
        "Loyal_Customer": (1.0, 0.0),
        "Creditworthiness": (0.7455937133620327, 8.887382419041861),
        "Cr_Util_Rate": (0.3659547645100451, 1.7149103774907906),
    }

    allow = st.button("Predict")

    if allow:
        """
        Try to upload the model and data here and make a fit,train here.
        then call just like how you did in the jupyter notebook
        """
        #X = np.array([gen, age, edu, inactive, stat, client_history, inc, transaction])
        #X = X.reshape(1, -1)
        #churn = xgb_model.predict(X.reshape(1, -1))
        #churn = xgb_model.predict(xgb.DMatrix(X.flatten()))
        client_data = {}

        # generate random values for each feature
        for feature, range_ in feature_ranges.items():
            if isinstance(range_, tuple):
                client_data[feature] = np.random.uniform(range_[0], range_[1])
            else:
                client_data[feature] = range_

        # convert the dictionary to a numpy array
        client_data = np.array(list(client_data.values()))

        # reshape the array to 2D
        client_data = client_data.reshape(1, -1)

        client_data_df = pd.DataFrame(client_data, columns=feature_ranges.keys())

        # make the prediction
        churn = xgb_model.predict(xgb.DMatrix(client_data.reshape(1, -1)))


        if churn[0] == 1:
            possibility = "will leave"
        if churn[0] == 0:
            possibility = "will not leave"
        st.subheader(f"The client is ${possibility} the credit card service.")