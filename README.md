# Client Churn Prediction

In this project, the goal is to analyze bank customer data for a credit card service and predict whether or not a customer will churn. 

The project begins by collecting and preprocessing the customer data from the bank's records. This includes cleaning and formatting the data, and handling any missing or invalid values. Next, the data is segmented using a custom method that takes into account various factors such as customer demographics, transaction history, and account usage. This allows for a more in-depth understanding of the customer base and their behavior.

After segmentation, the K-means clustering algorithm is applied to the data to further represent the distribution of customers. This allows for identifying patterns and trends within the customer segments, which can be useful in targeting retention efforts.

Finally, the XGBoost algorithm is used to train and test a model that can predict whether or not a customer is likely to churn. The model is trained on a portion of the data, and its performance is evaluated using metrics such as accuracy, precision, and recall. The model is then applied to the remaining data to make churn predictions.

The results of this project can be used by the bank to improve their retention efforts by targeting their marketing and retention strategies to specific segments of customers who are at a higher risk of churning. Additionally, this analysis can also help in identifying the factors that contribute the most to customer churn and addressing them.


## Demo

[Client Churn Prediction](https://churn-ml-capstone-project.streamlit.app/)


## Lessons Learned

Data Cleaning: We realized that gaining access to high-quality and clean data is one of the most important and challenging tasks. Missing or invalid data can lead to inaccurate or unreliable results. To overcome this, we made sure that the data was cleaned and transferred correctly.

Feature Engineering: We learned that the success of a predictive model is heavily dependent on the features used. Extracting relevant features from the data and creating new ones that capture the underlying patterns can be a challenging task. We used different feature selection and extraction methods like customized segmentation to overcome this challenge.To build financial attributes, we explored banks' approach to customer data and how they process it.

Model Selection and Tuning: There are many different machine learning models to choose from, and each has its own strengths and weaknesses. In addition, many models have several hyperparameters that need to be tuned to achieve optimal performance. To overcome this, we use various model selection and tuning techniques, such as cross-validation, to find the best model and its optimal parameters. 

Handling Imbalanced Data: The bank customer dataset is imbalanced, with a majority of the data points belonging to one class. This can lead to biased models that perform poorly on the minority class. To overcome this, we used the technique SMOTE to generate synthetic data.

Interpretability: In some cases, it can be challenging to understand and interpret the results of a predictive model, especially when working with complex models like XGBoost. To overcome this, one can use techniques such as feature importance, partial dependence plots, and SHAP values to gain insight into the model's decision making process.



## Authors

- [@melisnv](https://github.com/melisnv)
- [@didemp](https://github.com/didemp)


  
