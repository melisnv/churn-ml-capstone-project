import pandas as pd
import streamlit as st
import xgboost as xgb
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def read_data(data_path):
    data = pd.read_csv(data_path)

    return data


def split_data(data, target_feature):
    X = data.iloc[:, :-1]
    y = data[target_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, X_test, y_train, y_test


def xgboost_model(X_train, y_train):
    # make model
    #xgb_model = xgb.XGBClassifier(use_label_encoder=False, n_jobs=-1, eval_metric='mlogloss')
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    return xgb_model


def model_prediction(xgb_model, X_test):
    y_pred = xgb_model.predict(X_test)
    return y_pred


def show_predict_page():
    st.title("Client Churn Prediction")
    st.write("""#### Fill the information to predict whether client will leave""")

    st.title("")

    gender = ('Female', 'Male')

    education = ('Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate')

    status = ('Single', 'Married', 'Divorced')
    income = ('Less than $40K', '$40K - $60K', '$80K - $120K', '$60K - $80K', '$120K +')

    credit_limit = st.slider("Credit Limit", int(round(1000.0)), int(round(34516.0)), int(round(2000.0)))

    gen = st.selectbox("Gender", gender)
    gen_map = {"Female": 0, "Male": 1}
    gen = gen_map[gen]

    age = st.slider("Age", 18, 100, 25)
    age_dict = {(25, 35): 0, (35, 45): 1, (45, 55): 2, (55, 65): 3, (65, float('inf')): 4}
    for age_range, age_group in age_dict.items():
        if age_range[0] <= age <= age_range[1]:
            age = age_group
            break

    revol_balance = st.slider("Total Revolving Balance", 0, 5000, 1000)

    edu = st.selectbox("Education Level", education)
    edu_map = {'Uneducated': 0, 'High School': 0, 'Graduate': 1, 'College': 1, 'Post-Graduate': 1, 'Doctorate': 2}
    edu = edu_map[edu]

    inactive = st.slider("Inactivity", 0, 100, 25)

    stat = st.selectbox("Status", status)
    stat_map = {'Single': 0, 'Married': 1, 'Divorced': 0}
    stat = stat_map[stat]

    transaction = st.slider("Total Transaction", 0, int(round(18484.000000)), int(round(510.000000)))

    client_history = st.slider("Months as Client", 13, 60, 2)
    history_dict = {(0, 25): 0, (25, 35): 1, (35, 45): 2, (45, 60): 3, (60, float('inf')): 4}
    for hist_range, hist_group in history_dict.items():
        if hist_range[0] <= client_history <= hist_range[1]:
            client_history = hist_group
            break


    avg_open_to_buy = st.slider("Avg Open to Buy", 0, 50000, 15000)

    inc = st.selectbox("Income", income)
    inc_map = {'Less than $40K': 1, '$40K - $60K': 2, '$80K - $120K': 4, '$60K - $80K': 3, '$120K +': 5}
    inc = inc_map[inc]

    # creadiworthness
    approx_income = {1: 20000, 2: 50000, 4: 100000, 3: 70000, 5: 130000}
    app_income = approx_income[inc]
    creadiworthness = app_income / credit_limit


    amt_chng_q4_q1 = st.slider("Total Amount Change Q4-Q1", 0, 4, 2)

    random_features = {
        "Dependent_count": (0, 5),
        "Card_Category": (0, 3),
        "Total_Relationship_Count": (1, 6),
        "Contacts_Count_12_mon": (0, 6),
        "Total_Trans_Ct": (10, 140),
        "Total_Ct_Chng_Q4_Q1": (0, 4),
        "Avg_Utilization_Ratio": (0.0, 0.9),
        "Transaction_Count_per_Contact": (0.0, 139.0),
        "Loyal_Customer": (1, 0),
        "Cr_Util_Rate": (0, 99.9),
    }

    allow = st.button("Predict")

    if allow:

        # generate random values for each feature
        random_client_data = {}
        for feature, range_ in random_features.items():
            if range_ is None or len(range_) == 0:
                random_client_data[feature] = None
            else:
                random_client_data[feature] = np.random.choice(range_)

        user_input_features = {"Customer_Age": age, "Gender": gen, "Education_Level": edu, "Marital_Status": stat,
                               "Income_Category": inc, "Months_on_book": client_history,
                               "Months_Inactive_12_mon": inactive,
                               "Credit_Limit": credit_limit, "Total_Trans_Amt": transaction,
                               "Total_Amt_Chng_Q4_Q1":amt_chng_q4_q1,"Total_Revolving_Bal": revol_balance,
                               "Creditworthiness": creadiworthness,"Avg_Open_To_Buy": avg_open_to_buy}
        user_input_client_data = {}
        for feature, value in user_input_features.items():
            user_input_client_data[feature] = value

        # convert the dictionary to a numpy array
        client_data = {}
        client_data.update(random_client_data)
        client_data.update(user_input_client_data)
        client_data = np.array(list(client_data.values()))

        #df1 = pd.DataFrame(random_features, columns=random_features.keys()).assign(**user_input_features)
        #st.dataframe(df1)

        # convert dictionaries to dataframes
        #df1 = pd.DataFrame(random_features, columns=random_features.keys())
        #df2 = pd.DataFrame(user_input_features, columns=user_input_features.keys())

        # concatenate dataframes along columns axis
        #result = df1.join(df2)
        df1 = pd.DataFrame.from_dict(random_client_data, orient='index').T
        df2 = pd.DataFrame.from_dict(user_input_client_data, orient='index').T
        result = pd.concat([df1, df2], axis=1)
        st.dataframe(result)

        # reshape the array to 2D
        client_data = client_data.reshape(1, -1)

        #data = pd.read_csv('./data/unscaled_data.csv')
        unscaled_data_csv = Path(__file__).parents[1] / 'data/unscaled_data.csv'
        data = read_data(unscaled_data_csv)

        X_train, X_test, y_train, y_test = split_data(data, "Attrition_Flag")

        # train model
        xgb_model = xgboost_model(X_train, y_train)

        # make a new client prediction
        churn = model_prediction(xgb_model, client_data.reshape(1, -1))

        if churn[0] == 1:
            possibility = "will leave"
        if churn[0] == 0:
            possibility = "will not leave"
        st.subheader(f"The client {possibility} the credit card service.")
