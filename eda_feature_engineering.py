# libraries
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
sns.set_theme(style='white')

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler, RobustScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
sns.set(rc={"figure.figsize":(6,8)})


def read_data(file_path):
    data = pd.read_csv(file_path)
    # remove first column (customer number) and last two columns
    return data


def examine_data(data):
    return display(data.describe().T)


def cat_data(data):
    categorical_data = list(data.select_dtypes('object'))
    return categorical_data


def num_data(data):
    numerical_data = list(data.select_dtypes(['int64','float64']))
    return numerical_data


def visualize_countplot(data, feature_name, title, x_label, file_name):
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    sns.countplot(x=data[feature_name],
                  palette=['#FFE9AE', '#EAC7C7', '#B08BBB', '#9E7676', '#B2C8DF', '#C8DBBE'])
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()


def visualize_histplot(data, feature_name, title, x_label, file_name,hue=None):
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})
    if hue:
        sns.histplot(data, x=feature_name, kde=True, color="#6368af",hue=hue)
    else:
        sns.histplot(data, x=feature_name, kde=True, color="#6368af")
    plt.xlabel(x_label)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()


def visualize_heatmap(data, feature_list, title, file_name):
    correlation = data[feature_list].corr()
    colormap = sns.color_palette("Greens")
    plt.figure(figsize=(8, 10))
    sns.heatmap(correlation, annot=True, cmap=colormap)
    plt.title(title)
    plt.savefig(file_name)
    plt.show()


def visualize_piechart(data, feature_name, title, file_name,labels=False):
    labels = list(data[feature_name].unique())
    data.groupby(feature_name).size().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 20, 'color': "w"},
                                           colors=['#6B7AA1', '#11324D'],labels=labels)
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    plt.title(title)
    plt.ylabel("")
    plt.savefig(file_name)
    plt.show()


def visualize_boxplot(data, feature_name_x,feature_name_y, title, file_name):
    sns.boxplot(x=data[feature_name_x], y=data[feature_name_y],
                palette=['#D8AC9C', '#34626C', '#8DB596', '#EFD9D1', '#92817A', '#BBBBBB', '#F7DAD9'])
    plt.title(title)
    plt.savefig(file_name)
    plt.show()


def call_visualization(data):
    visualize_countplot(data, 'Attrition_Flag', 'Distribution of Churn Customer',
                        'customer state', 'customer_count.png')
    visualize_histplot(data, 'Months_on_book', 'Distribution of Customer\'s Relationship',
                       'relationships in month', 'customer_dist.png')
    visualize_histplot(data, 'Customer_Age', 'Distribution of Customer\'s Age',
                       'customer age', 'age_dist.png')
    visualize_countplot(data, 'Total_Relationship_Count', 'The Number of Products Held by Customer',
                        'number of products held by customer', 'product_count.png')
    visualize_histplot(data, 'Total_Revolving_Bal', 'Distribution of Total Revolving Balance on the Credit Card',
                       'balance on credit card', 'revolbalance_dist.png')
    visualize_heatmap(data, ["Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
                             "Total_Ct_Chng_Q4_Q1", "Attrition_Flag"], "Transaction Correlation",
                      "transaction_heat.png")
    visualize_piechart(data, "Gender", "Distribution of Gender", "gender_piechart.png")
    visualize_histplot(data, "Education_Level", "Distribution of Customer\'s Education Level and Churn",
                       "education level", "ed_lvl_churn.png", hue="Attrition_Flag")
    visualize_piechart(data, "Marital_Status", "Distribution of Marital Status", "marital_piechart.png")
    visualize_countplot(data, 'Income_Category', 'The Annual Income of the Client',
                        'amount of annual income', 'income_count.png')
    visualize_boxplot(data, "Income_Category", "Credit_Limit", "Income Category vs Credit Limit",
                      "inccat_crlmt_boxplot.png")


def map_categories(data, feature, category_dict):
    data[feature] = data[feature].apply(lambda x: category_dict.get(x, np.nan))
    data[feature] = data[feature].fillna(data[feature].median())
    data[feature] = data[feature].astype(int)
    return data


def data_process_feature_engineering(data):
    # attrition flag
    data["Attrition_Flag"] = data["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})

    # customer age
    data.loc[(data['Customer_Age'] >= 25) & (data['Customer_Age'] < 35), 'Customer_Age'] = 0
    data.loc[(data['Customer_Age'] >= 35) & (data['Customer_Age'] < 45), 'Customer_Age'] = 1
    data.loc[(data['Customer_Age'] >= 45) & (data['Customer_Age'] < 55), 'Customer_Age'] = 2
    data.loc[(data['Customer_Age'] >= 55) & (data['Customer_Age'] < 65), 'Customer_Age'] = 3
    data.loc[(data['Customer_Age'] >= 65), 'Customer_Age'] = 4
    data['Customer_Age'] = data['Customer_Age'].astype(int)

    # customer's relationship in months
    data.loc[(data['Months_on_book'] <= 25), 'Months_on_book'] = 0
    data.loc[(data['Months_on_book'] > 25) & (data['Months_on_book'] <= 35), 'Months_on_book'] = 1
    data.loc[(data['Months_on_book'] > 35) & (data['Months_on_book'] <= 45), 'Months_on_book'] = 2
    data.loc[(data['Months_on_book'] > 45) & (data['Months_on_book'] <= 60), 'Months_on_book'] = 3
    data['Months_on_book'] = data['Months_on_book'].astype(int)

    # creating a new feature
    data["Transaction_Count_per_Contact"] = data["Total_Trans_Ct"] / data["Months_on_book"]
    data["Transaction_Count_per_Contact"].replace([np.inf, -np.inf], 0, inplace=True)

    # gender
    data.loc[(data['Gender'] == "M"), 'Gender'] = 0
    data.loc[(data['Gender'] == "F"), 'Gender'] = 1

    # Education
    ed_cats = {'Uneducated': 0, 'High School': 0, 'Graduate': 1, 'College': 1, 'Post-Graduate': 1,
               'Unknown': np.nan, 'Doctorate': 2}
    data = map_categories(data, 'Education_Level', ed_cats)

    # Marital_Status
    marital_cats = {'Single': 0, 'Married': 1, 'Divorced': 0, 'Unknown': np.nan}
    data = map_categories(data, 'Marital_Status', marital_cats)

    # Income_Category
    income_cats = {'Less than $40K': 1, '$40K - $60K': 2, '$80K - $120K': 4, '$60K - $80K': 3,
                   'Unknown': np.nan, '$120K +': 5}
    data = map_categories(data, 'Income_Category', income_cats)

    # Card_Category_Category
    card_cats = {'Blue': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
    data = map_categories(data, 'Card_Category', card_cats)

    # New loyal customer feature
    data["Loyal_Customer"] = 0
    data.loc[(data["Months_on_book"] > 1) & (data["Months_Inactive_12_mon"] < 4), "Loyal_Customer"] = 1

    # Creadithorthiness
    approx_income = {1: 20000, 2: 50000, 4: 100000, 3: 70000, 5: 130000}
    data['Approx_Income'] = data['Income_Category'].apply(lambda x: approx_income[x])
    data['Creditworthiness'] = data['Approx_Income'] / data['Credit_Limit']
    data.drop('Approx_Income', axis=1,inplace = True)

    # Credit utilization rate
    data["Cr_Util_Rate"] = data['Total_Revolving_Bal'] / data['Credit_Limit']

    return data


def scale_data(data,target_feature):
    robust_scaler = RobustScaler()
    robust_scaler.fit_transform(data.drop(target_feature, axis=1))
    scaled_data = robust_scaler.transform(data.drop(target_feature, axis=1))
    copy_data = data.drop(data.columns[0], axis=1)
    scaled_data = pd.DataFrame(scaled_data, columns=copy_data.columns)

    return scaled_data


def save_data(data):
    return data.to_csv("customer_data.csv", index=False)


def main(data_path, visualization=False):
    # read data
    data = pd.read_csv(data_path)
    data = data.iloc[:, 1:-2] # remove unnecessary columns

    # examining the data with statistics
    examine_data(data)

    # divide data to make observations
    categorical_data = cat_data(data)
    numerical_data = num_data(data)

    if visualization:
        # save important visualisations to inspect data
        call_visualization(data)

    # feature engineering
    processed_data = data_process_feature_engineering(data)

    # scaling the data
    scaled_data = scale_data(data, "Attrition_Flag")
    scaled_data["Attrition_Flag"] = data["Attrition_Flag"]

    # save processed data
    save_data(scaled_data)



# main('./data/BankChurners.csv', visualization=True)
main('./data/BankChurners.csv')