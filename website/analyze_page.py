import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


def read_data(file_path):
    data = pd.read_csv(file_path)
    # remove first column (customer number) and last two columns
    return data


def map_categories(data, feature, category_dict):
    data[feature] = data[feature].apply(lambda x: category_dict.get(x, np.nan))
    data[feature] = data[feature].fillna(data[feature].median())
    data[feature] = data[feature].astype(int)
    return data


#@st.cache(allow_output_mutation=True)
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
    data.drop('Approx_Income', axis=1, inplace=True)

    # Credit utilization rate
    data["Cr_Util_Rate"] = data['Total_Revolving_Bal'] / data['Credit_Limit']

    return data



#processed_data = data_process_feature_engineering(data)
#processed_data = processed_data.drop("CLIENTNUM",axis=1)


def show_analysis():
    st.title("Analyze of the Credit Card Customer Data")
    st.title("")

    data_path = "./data/BankChurners.csv"
    data = read_data(data_path)
    #data = data.iloc[:, 1:-2]

    st.subheader("Data")
    st.dataframe(data.iloc[:, 1:-2].head())

    st.subheader("Distribution of Customer\'s Age")
    fig1 = plt.figure(figsize=(10, 4))
    plt.xlabel("customer age")
    sns.histplot(data, x="Customer_Age", kde=True, color="#6368af")
    st.pyplot(fig1)

    st.title("")

    st.subheader("Distribution of Customer\'s Age and Income")
    fig2 = plt.figure(figsize=(10, 4))
    plt.xlabel("customer age")
    sns.histplot(data, x="Customer_Age", kde=True, hue="Income_Category", color=['#F7DBF0', '#BCCC9A', '#CDBBA7',
                                                                                 '#716F81', '#A0937D', '#A58FAA'])
    st.pyplot(fig2)

    st.markdown("As can be seen from this chart, it is common for younger customers to earn less than $40K. A similar "
                "pattern is also seen in middle-aged customers. The highest income, 80K and above, is mostly seen in "
                "customers between the ages of 45-55.")

    st.title("")

    st.subheader("Distribution of Customer's Relationship")
    fig3 = plt.figure(figsize=(10, 4))
    plt.xlabel("relationships in month")
    sns.histplot(data, x='Months_on_book', kde=True, color="#694E4E")
    st.pyplot(fig3)

    st.markdown("It is possible that the pick at 36 months for the '*Months_on_book*' feature is due to a specific "
                "policy or event that occurred within the bank. **36 months** is not an inherently significant number "
                "in banking, but it could be a duration that aligns with certain policies or promotions that the bank "
                "has implemented. For example, it could be that the bank has a promotion for customers who have been "
                "with them for at least 36 months, which would incentivize customers to stay with the bank for that "
                "duration. Without more information about the bank's policies and events, it's difficult to say for "
                "certain why there is a peak at 36 months.It is also possible that 36 months is a common duration for "
                "customers to stay with a bank, either due to the nature of the services they provide or the "
                "customer's own financial needs. It could also be due to certain policies or regulations that limit "
                "or encourage certain behaviors by the bank or its customers.")

    st.title("")

    st.subheader("Distribution of Customer\'s Relationship and Churn State")
    fig4 = plt.figure(figsize=(10, 4))
    plt.xlabel("relationship scale")
    sns.histplot(data, x='Months_on_book', kde=True, color=['#BCCC9A','#FFC898','#B5DEFF','#D57E7E'],hue=data["Attrition_Flag"])
    st.pyplot(fig4)
    st.markdown("As can be seen here, the highest non-churn customer value is seen between 35-45 months. In terms of "
                "customer relationship, this corresponds to a relationship of 3-4 years. The reason for this may be "
                "that customers who have a credit card membership between 35-45 months, as mentioned before, "
                "may terminate their membership after completing the required period in the membership agreement.")

    st.title("")

    st.subheader("The Number of Times the Client Made a Contact with the Bank")
    st.markdown("This feature represents the number of times the client reached to bank one time and the same vice "
                "versa. This feature could be a useful feature to include in the analysis as it provides insight into "
                "the level of engagement and communication between the bank and its customers.")
    fig5 = plt.figure(figsize=(10, 4))
    c = sns.countplot(x=data["Contacts_Count_12_mon"],palette=['#FFE9AE', '#EAC7C7', '#B08BBB', '#9E7676', '#B2C8DF', '#C8DBBE'])
    c.set_xlabel("number of contacts by customer")
    st.pyplot(fig5)

    st.title("")

    st.subheader("Distribution of Customer Credit Limit")
    fig6 = plt.figure(figsize=(10, 4))
    sns.histplot(data, x='Credit_Limit', kde=True, color="#D77FA1")
    plt.xlabel("limits in total")
    st.pyplot(fig6)
    st.markdown("As can be seen here, users with a credit card limit of approximately 35000 show a high value. When "
                "the patterns of these users are examined: It is seen that all of the customers have a credit limit "
                "of 34516.0. In general, it is seen that the number of inactive months is low, and the income "
                "categories are 60K-80K, 80K-120K and $120K +.This credit limit can be a special credit limit that "
                "the bank gives to its customers who meet certain criteria. At the same time, there may be a limit "
                "that it offers to its customers as a promotion. In addition, when the customers with this limit are "
                "examined for how many months they have been customers of the bank, it is seen that the distribution "
                "is between 1-3 months. This shows that customers are given a high limit credit card membership by "
                "applying a certain promotion. An important policy of the bank may have been to build a high-income "
                "client base.")

    st.title("")

    st.subheader("Distribution of Customer Total Revolving Balance")
    fig7 = plt.figure(figsize=(10, 4))
    sns.histplot(data, x='Total_Revolving_Bal', kde=True, color="#7F7C82")
    plt.xlabel("revolving balance in total")
    st.pyplot(fig7)
    st.markdown("This feature represents the total revolving balance on the credit card of the customer, it could be "
                "a useful feature in the analysis as it provides insight into the customer's credit utilization and "
                "spending habits. It is important to keep in mind that, this feature could be correlated with other features "
                "and could be impacted by external factors such as interest rates, economic conditions, etc. It would "
                "be useful to look at the bank's internal data and external factors that might have led to this "
                "feature and also keep in mind that, this feature alone cannot be used for any predictions or "
                "classifications because it is a continuous variable.")

    st.title("")

    st.subheader("Correlation")
    st.markdown("Total_Trans_Ct and Total_Trans_Amt are highly correlated.")
    fig8 = plt.figure(figsize=(10, 4))
    correlation = data[["Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
                        "Total_Ct_Chng_Q4_Q1", "Attrition_Flag"]].corr()
    colormap = sns.color_palette("Blues")
    sns.heatmap(correlation, annot=True, cmap=colormap)
    st.pyplot(fig8)

    st.title("")

    st.subheader("Distribution of Gender")
    fig9 = plt.figure(figsize=(5, 3))
    data.groupby('Gender').size().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 5, 'color': "w"},
                                       colors=['#6B7AA1', '#11324D'])
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    plt.ylabel("")
    st.pyplot(fig9)

    st.title("")

    st.subheader("Distribution of Customer\'s Education Level and Churn")
    fig10 = plt.figure(figsize=(10, 4))
    sns.histplot(data, x="Education_Level", kde=True, palette=['#9E7777','#6F4C5B'], hue="Attrition_Flag")
    plt.xlabel("education level")
    st.pyplot(fig10)

    st.title("")

    st.subheader("Distribution of Marital Status")
    fig11 = plt.figure(figsize=(10, 4))
    labels = list(data["Marital_Status"].unique())
    data.groupby('Marital_Status').size().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 5, 'color': "w"},
                                               labels=labels, colors=['#C1AC95', '#B5CDA3', '#3A6351', '#5F939A'])
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    plt.ylabel("")
    st.pyplot(fig11)

    st.title("")

    st.subheader("The Annual Income of the Client")
    fig12 = plt.figure(figsize=(10, 4))
    sns.countplot(x=data["Income_Category"], palette=['#6E85B7', '#B2C8DF', '#C9BBCF', '#898AA6', '#90C8AC', '#73A9AD',
                                                      '#525E75'])
    plt.xlabel("amount of annual income")
    st.pyplot(fig12)
    st.markdown("As can be seen from this chart, a large proportion of customers have an income of less than 40K. "
                "This rate is followed by the 40K-60K income group with approximately 2000 customers.")

    st.title("")

    st.subheader("The Annual Income of the Client with Credit Limit")
    fig13 = plt.figure(figsize=(10, 4))
    sns.boxplot(x=data["Income_Category"], y=data["Credit_Limit"],
                palette=['#D8AC9C', '#34626C', '#8DB596', '#EFD9D1', '#92817A', '#BBBBBB', '#F7DAD9'])
    plt.xlabel("amount of annual income")
    plt.ylabel("credit limit")
    st.pyplot(fig13)
    st.markdown("As can be seen from this analysis, the customer with the highest credit limit is in the 80K-120K and "
                "120K+ income groups. While the general limit average of the 120K income group is close to 20000, "
                "the limit average of the 80K-120K income group is close to 15000. There are especially 60K-80 and "
                "unknown income group customers with high limits, but the average limit for both groups is around "
                "10000. Although the average credit score of the lowest income group is low as expected, "
                "some customers seem to have a limit of up to 17000.")

    st.title("")

    st.subheader("The Card Type")
    fig14 = plt.figure(figsize=(10, 4))
    sns.countplot(x=data["Card_Category"], palette=['#6E85B7', '#B2C8DF', '#C9BBCF', '#898AA6', '#90C8AC', '#73A9AD',
                                                    '#525E75'])
    plt.xlabel("card types")
    st.pyplot(fig14)
    st.markdown("In the context of credit cards, 'Blue', 'Silver', 'Gold', and 'Platinum' are typically used to "
                "indicate the level of benefits and rewards offered by the card. A Blue card is often a basic card "
                "with standard features and limited rewards. A Silver card may offer slightly more benefits and "
                "rewards than a Blue card.A Gold card typically offers a higher level of benefits and rewards, "
                "such as cashback or travel rewards. A Platinum card is usually the highest level of card offered, "
                "with the most extensive benefits and rewards, such as concierge service and exclusive access to "
                "events. The specific benefits and rewards associated with each card level can vary depending on the "
                "bank and the card issuer. Some banks also offer different levels of cards such as Signature, World, "
                "Elite etc. But generally, these four levels of cards are the most common.")

    st.title("")

    st.subheader("Distribution of Customer\'s Card Types and Income Groups")
    fig15 = plt.figure(figsize=(10, 4))
    sns.histplot(data, x="Card_Category", kde=True, hue="Income_Category")
    plt.xlabel("card types")
    st.pyplot(fig15)

    st.title("")
    st.title("New Features")

    st.subheader("Distribution of Creditworthiness of the Customer")
    st.markdown("This feature could be useful because it gives an indication of how much credit a customer has "
                "relative to their income. This could be an indicator of creditworthiness and the risk of default. "
                "The threshold for the creditworthiness can vary depending on the bank's policies and the specific "
                "credit product. In general, a lower ratio indicates that a customer has a higher income relative to "
                "their credit limit, which is considered a positive indicator of creditworthiness. Conversely, "
                "a higher ratio indicates that a customer has a lower income relative to their credit limit, "
                "which is considered a negative indicator of creditworthiness. Banks typically use a variety of "
                "factors to assess a customer's creditworthiness, including their income, credit score, "
                "credit history, and debt-to-income ratio. The exact threshold for each of these factors can also "
                "vary depending on the bank's policies and the specific credit product. For example, for a mortgage "
                "loan, banks generally look for a debt-to-income ratio below 43%, while for a credit card, "
                "the threshold may be higher. However, it's important to keep in mind that the ratio alone may not be "
                "enough to determine creditworthiness. Banks generally use a variety of factors, such as credit "
                "score, credit history, and debt-to-income ratio, to make lending decisions. Therefore we will create "
                "another feature to calculate credit utilization rate by making a ratio between Total_Revolving_Bal "
                "and Credit_Limit.")
    code_crediworthness = '''approx_income = {1: 20000, 2: 50000, 4: 100000, 3:70000, 5: 130000}
    data['Approx_Income'] = data['Income_Category'].apply(lambda x: approx_income[x])'''
    st.code(code_crediworthness, language="python")

    processed_data = data_process_feature_engineering(data)

    st.markdown("For example, for the group '40K', we can assign the value 40,000. For the group '60K-80K', "
                "we can assign the value 70,000 (midpoint of the range). And for the group '80K-120K' we can assign "
                "the value 100,000. Keep in mind that this is just an example, and the values we choose will depend "
                "on the data and what makes sense for the analysis. Once we've assigned numerical values to the "
                "income groups, we can calculate the Creditworthiness feature by dividing the numerical income value "
                "by the Credit_Limit value."
                ""
                "Once we've assigned numerical values to the income groups, we can calculate the Creditworthiness "
                "feature by dividing the numerical income value by the Credit_Limit value.")
    code_crediworthness2 = '''data['Creditworthiness'] = data['Approx_Income'] / data['Credit_Limit']'''
    st.code(code_crediworthness2, language="python")
    fig16 = plt.figure(figsize=(10, 4))
    sns.histplot(processed_data, x='Creditworthiness', kde=True, color="#8B7E74")
    plt.xlabel("creditworthiness ratio")
    st.pyplot(fig16)
    st.markdown("In general, there is customer data with low income and high credit limit. Most of them have the same "
                "number of credit card (bank) participation months, that is, to attract customers with a promotion.")

    st.title("")

    st.subheader("Loyal Customer")
    st.markdown("Earning loyalty is critical for banks, as it pays off with higher revenues, a lower cost to serve "
                "and happier employees. "
                ""
                "Establish a relationship between the Months_on_book and Months_Inactive_12_mon attributes and create "
                "a new column named loyal customer. Because a loyal customer will be less likely to churn.")
    code_loyalcustomer = '''data.loc[(data["Months_on_book"] > 1) & (data["Months_Inactive_12_mon"] < 4),
    "Loyal_Customer"] = 1 '''
    st.code(code_loyalcustomer, language="python")
    st.markdown("In general, there is customer data with low income and high credit limit. Most of them have the same "
                "number of credit card (bank) participation months, that is, to attract customers with a promotion.")

    st.title("")

    st.subheader("Credit Utilization Rate")
    st.markdown("Credit utilization rate is the ratio of a borrower's outstanding credit card balances to their "
                "credit limits. Banks use this ratio to determine a borrower's creditworthiness. A lower credit "
                "utilization rate indicates that a borrower is using less of their available credit and is considered "
                "more financially responsible, while a higher credit utilization rate may indicate that the borrower "
                "is overextending themselves and may be at a higher risk of default. A general rule of thumb is to "
                "keep the credit utilization rate below 30%. It's also worth noting that credit utilization rate is "
                "one of the factors that can affect the credit score, a lower credit utilization rate is better for "
                "credit score.")
    code_loyalcustomer = '''data["Cr_Util_Rate"] = data['Total_Revolving_Bal'] / data['Credit_Limit']'''
    st.code(code_loyalcustomer, language="python")
    st.write(processed_data["Cr_Util_Rate"].describe())
    st.markdown("The mean value of 0.27 suggests that on average, the customers are utilizing about 27% of their "
                "credit limit. This is generally considered to be a relatively low utilization rate, which is good "
                "for credit scores and implies that customers are managing their credit well. "
                "The standard deviation of 0.28 suggests that there is a relatively high degree of variation in "
                "credit utilization rates among the customers. The minimum value of 0.0 suggests that some customers "
                "are not utilizing any of their credit limit, while the maximum value of 0.99 suggests that some "
                "customers are utilizing almost all of their credit limit. "
                "To analyze this feature to predict customer churn, one way is to create a binary variable indicating "
                "whether or not each customer's credit utilization rate is above or below a certain threshold. For "
                "example,could create a new variable that is 1 if the customer's utilization rate is above 30%, "
                "and 0 if it is below 30%. Then could use this variable as a feature in the machine learning model to "
                "predict churn.")


    #processed_data.drop("CLIENTNUM",axis=1,inplace=True)
    st.write(processed_data.head())