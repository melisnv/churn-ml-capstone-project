import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='white')
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
sns.set(rc={"figure.figsize":(6,8)})

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


import xgboost as xgb
from xgboost import plot_importance


def read_data(data_path):
    data = pd.read_csv(data_path)

    return data


def split_data(data, target_feature):
    X = data.iloc[:, :-1]
    y = data[target_feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, X_test, y_train, y_test


def over_sample(X_train, y_train):
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train


def xgboost_model(X_train, y_train):
    # make model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, n_jobs=-1,eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model


def model_prediction(xgb_model, X_test):
    y_pred = xgb_model.predict(X_test)
    return y_pred


def evaluation_metrics(xgb_model, X_test, y_test, y_pred):
    print("Model Score: ")
    print(xgb_model.score(X_test, y_test))
    print("\n")
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))
    print("\n")
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))


def visualize_evaluation(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    colormap = sns.color_palette("Blues")
    plt.figure(figsize=(12, 15));
    sns.heatmap(cf_matrix, annot=True, cmap=colormap, fmt=".1f")
    plt.title("Confusion Matrix")
    plt.show()


def create_new_client():
    feature_ranges = {
        "Customer_Age": (-2.0, 2.0),
        "Gender": (-1.0, 0.0),
        "Dependent_count": (-1.0, 1.5),
        "Education_Level": (-1.0, 1.0),
        "Marital_Status": (-1.0, 1.0),
        "Income_Category": (-0.5, 1.5),
        "Card_Category": (0.0, 3.0),
        "Months_on_book": (-2.0, 1.0),
        "Total_Relationship_Count": (-1.5, 1.0),
        "Months_Inactive_12_mon": (-2.0, 4.0),
        "Contacts_Count_12_mon": (-2.0, 4.0),
        "Credit_Limit": (-0.3654273127753303, 3.520352422907489),
        "Total_Revolving_Bal": (-0.895438596491228, 0.8708771929824561),
        "Avg_Open_To_Buy": (-0.4067022086824067, 3.6372370964907135),
        "Total_Amt_Chng_Q4_Q1": (-3.2280701754385968, 11.671052631578949),
        "Total_Trans_Amt": (-1.3107716109069811, 5.641075227228776),
        "Total_Trans_Ct": (-1.5833333333333333, 2.0),
        "Total_Ct_Chng_Q4_Q1": (-2.9745762711864407, 12.76271186440678),
        "Avg_Utilization_Ratio": (-0.3666666666666666, 1.7145833333333331),
        "Transaction_Count_per_Contact": (-1.2777777777777777, 3.87037037037037),
        "Loyal_Customer": (-1.0, 0.0),
        "Creditworthiness": (-0.7455937133620327, 8.887382419041861),
        "Cr_Util_Rate": (-0.3659547645100451, 1.7149103774907906),
    }

    client_data = {}

    # generate random values for each feature
    for feature, range_ in feature_ranges.items():
        if isinstance(range_, tuple):
            client_data[feature] = np.random.uniform(range_[0], range_[1])
        else:
            client_data[feature] = np.random.choice(range_)

    # convert the dictionary to a numpy array
    client_data = np.array(list(client_data.values()))

    # reshape the array to 2D
    client_data = client_data.reshape(1, -1)
    client_data_df = pd.DataFrame(client_data, columns=feature_ranges.keys())

    return client_data, client_data_df


def important_features(X_train, y_train):
    # https://stackoverflow.com/questions/40664776/how-to-change-size-of-plot-in-xgboost-plot-importance
    # Train the model
    xgbmodel = xgboost_model(X_train, y_train)

    # Plot feature importances
    fig, ax = plt.subplots(figsize=(12, 15))
    plot_importance(xgbmodel, ax=ax, color='#b7d3b3')
    plt.title("XGBClassifer Feature Importances");
    plt.show();


def main(data_path):
    data = read_data(data_path)
    print(data)
    X_train, X_test, y_train, y_test = split_data(data, "Attrition_Flag")
    X_train, y_train = over_sample(X_train, y_train)
    print(X_train.shape)
    print(y_train.shape)

    # construct model
    xgb_model = xgboost_model(X_train, y_train)
    # make predictions
    y_pred = model_prediction(xgb_model, X_test)
    # perform evaluation
    evaluation_metrics(xgb_model, X_test, y_test, y_pred)
    # visual confusion matrix
    visualize_evaluation(y_test, y_pred)

    # create new client to test the model
    client_data, client_data_df = create_new_client()
    client_pred = model_prediction(xgb_model, client_data.reshape(1, -1))

    important_features(X_train, y_train)

    if client_pred[0] == 1:
        print("Client will leave the credit card services.")
    else:
        print("Client will not leave the credit card services.")


main("./data/customer_data.csv")
