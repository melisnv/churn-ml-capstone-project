import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_data(data_path):
    data = pd.read_csv(data_path)
    data = data.drop([
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
        axis='columns')
    return data


def elbow_method(data, cols):
    cluster_num = []

    for i in range(2, 16):
        km = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=1)
        data_scaled = StandardScaler().fit_transform(data[cols])
        km.fit(data_scaled)
        cluster_num.append(km.inertia_)

    return cluster_num


@st.cache(allow_output_mutation=True)
def pipeline_process(df, cols):
    preprocessor = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2, random_state=1))])

    clusterer = Pipeline([("kmeans", KMeans(n_clusters=4, init="k-means++",
                                            n_init=50, max_iter=1000, random_state=1))])

    pipeline = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    pipeline.fit(df[cols])
    return pipeline


def show_cluster_page():
    df = read_data("../data/BankChurners.csv")

    st.title("Client Clustering with K-Means Algorithm")
    st.title("")

    st.markdown("K-means is a popular unsupervised machine learning algorithm for clustering data into K clusters. "
                "The basic idea behind the algorithm is to define clusters such that the distance between the data "
                "points within a cluster is minimized while the distance between the points in different clusters is "
                "maximized.")

    cols = df.iloc[:, 9:].select_dtypes(['uint8', 'int64', 'float64']).columns
    cluster_numbers = elbow_method(df, cols)

    fig1 = plt.figure(figsize=(10, 8))
    plt.plot(range(2, 16), cluster_numbers)
    plt.title('Elbow Method')
    plt.xlabel('K Clusters')
    plt.ylabel('Within-cluster Sums of Squares')
    st.pyplot(fig1)

    st.markdown("Principal component analysis (PCA) is a statistical technique that is used to identify patterns in "
                "data. It is often used as a preprocessing step before applying other techniques, such as k-means "
                "clustering. In the context of k-means clustering, PCA can be used to reduce the dimensionality of "
                "the data by projecting it onto a lower-dimensional space, which can make the k-means algorithm more "
                "computationally efficient and effective. "
                "By using PCA before applying k-means clustering, you can potentially improve the results of the "
                "clustering by removing noise and reducing the complexity of the data.")

    pipeline = pipeline_process(df, cols)
    preprocessed_data = pipeline["preprocessor"].transform(df[cols])
    predicted_labels = pipeline["clusterer"]["kmeans"].labels_
    st.write('Silhouette Score: ', round(silhouette_score(preprocessed_data, predicted_labels), 3))

    preprocessed_data = pd.DataFrame(pipeline["preprocessor"].transform(df[cols]),
                                     columns=['x', 'y'])

    preprocessed_data["predicted_labels"] = pipeline["clusterer"]["kmeans"].labels_  # predicted labels

    fig2 = plt.figure(figsize=(10, 10))
    sns.scatterplot(x="x",
                    y="y", data=preprocessed_data,
                    hue="predicted_labels",
                    palette=['#FFE9AE', '#EAC7C7', '#B08BBB',
                             '#9E7676'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("Clusters from Credit Card Users")
    st.pyplot(fig2)

    df["cluster"] = pipeline["clusterer"]["kmeans"].labels_
    st.dataframe(df.head(5))

    st.title("Cluster Distribution")
    fig3 = plt.figure(figsize=(10, 12))
    df.groupby('cluster').size().plot(kind='pie', autopct='%1.1f%%', textprops={'fontsize': 15, 'color': "w"},
                                      colors=['#6B7AA1', '#11324D', '#9E7777', '#6F4C5B'])
    plt.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    plt.ylabel("")
    st.pyplot(fig3)

    st.title("Clusters and Their Inactivity")
    fig4 = plt.figure(figsize=(12, 15))
    sns.boxplot(x=df["cluster"], y=df["Months_Inactive_12_mon"],
                palette=['#6E85B7', '#B2C8DF', '#C9BBCF', '#898AA6', '#90C8AC', '#73A9AD'])
    plt.xlabel("clusters")
    plt.ylabel("inactivity")
    st.pyplot(fig4)

    st.title("Clusters and Their Transactions")
    fig5 = plt.figure(figsize=(12, 15))
    sns.boxplot(x=df["cluster"], y=df["Total_Trans_Amt"],
                palette=['#90C8AC', '#92817A', '#BBBBBB', '#6E85B7'])
    plt.xlabel("clusters")
    plt.ylabel("transaction amount limit")
    st.pyplot(fig5)

    st.title("Clusters and Their Total Revolving Balance")
    fig6 = plt.figure(figsize=(12, 15))
    sns.boxplot(x=df["cluster"], y=df["Total_Revolving_Bal"],
                palette=['#FFE9AE', '#EAC7C7', '#B08BBB', '#9E7676', '#B2C8DF', '#C8DBBE'])
    plt.xlabel("clusters")
    plt.ylabel("total revolving balance")
    st.pyplot(fig6)

    st.title("Clusters and Their Income Category")
    fig7 = plt.figure(figsize=(12, 15))
    sns.countplot(x=df["cluster"], hue=df["Income_Category"],
                  palette=['#FFE9AE', '#EAC7C7', '#B08BBB', '#9E7676', '#B2C8DF', '#C8DBBE'])
    plt.xlabel("clusters")
    plt.ylabel("income category count")
    st.pyplot(fig7)

    st.title("Findings")
    st.markdown("The final solution produced four clusters: Cluster 0 (n = 1242), Cluster 1 (n = 3752), Cluster 2 (n "
                "= 4050), and Cluster 3 (n = 1083).")
    st.markdown("The silhouette score was 0.39, indicating moderate similarity within clusters and dissimilarity "
                "between clusters.")

    st.markdown("Cluster 0 was found to be composed mostly of the customers with low revolving balance and a low "
                "amount of "
                "total transaction. They also have an average loyalty to the credit card service and have a high "
                "income.")
    st.markdown("Cluster 1 was found to be composed mostly of the customers with highest revolving balance and an "
                "average amount of total transaction. They also have an average loyalty to the credit card service "
                "and lowest income.")
    st.markdown("Cluster 2 was found to be composed mostly of the customers with lowest revolving balance and a "
                "lowest amount of total transaction. They also have an average loyalty to the credit card service and "
                "comparable low income.")
    st.markdown("Cluster 3 was found to be composed mostly of the customers with high revolving balance and a "
                "higher amount of total transaction. They also have a higher loyalty to the credit card service and "
                "an average income.")

    st.title("Conclusion")
    st.markdown("The K-means clustering analysis successfully identified four clusters of customers with distinct "
                "characteristics.")
    st.markdown(
        "These clusters can be used to segment the customer base and target marketing efforts more effectively.")
    st.markdown("Additional analysis, such as association rule mining, can be performed on these clusters to uncover "
                "further insights.")
    st.markdown("Based on the clustering result, Cluster 3 is the most valuable customers and Cluster 1 and 2 are less "
                "valuable customers, the company should focus on retaining Cluster 0 and try to improve its "
                " loyalty as they have high income.")
