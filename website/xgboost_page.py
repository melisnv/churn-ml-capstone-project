import streamlit as st
from PIL import Image



def show_xgboost_page():
    st.title("Motivation Behind the Model")

    st.title("")

    st.markdown("XGBoost is a powerful and popular machine-learning algorithm that is widely used for classification "
                "and regression problems. It is particularly useful for large datasets or datasets with a large "
                "number of features.")

    st.markdown("One of the main reasons we choose XGBoost is its ability to handle missing values and categorical "
                "variables effectively. After our examination, we found out that the data is highly imbalanced. "
                "XGBoost is also known for its ability to handle imbalanced datasets, a dataset where the number of "
                "observations in one class is significantly different from the number of observations in another "
                "class. ")

    st.markdown("Another reason XGBoost is its robustness to overfitting and its ability to achieve higher accuracy "
                "than traditional gradient boosting machines. This is due to the regularization techniques such as L1 "
                "and L2 that are used. Additionally, XGBoost also offers parallel processing which allows for faster "
                "computation, making it more efficient and faster than other algorithms. It also has a built-in "
                "feature importance mechanism, which can help identify the most important features that contribute to "
                "the prediction.")

    st.markdown("We thought that all these features make XGBoost a powerful and versatile algorithm well-suited for "
                "customer churn prediction problems and come to the conclusion that it is also widely used in various "
                "industries such as finance, e-commerce, and healthcare.")

    st.title("The Steps Applied")
    st.title("")

    st.header("SMOTE")
    st.markdown("We realized that it affects the performance of the model as the dataset used is highly imbalanced "
                "and we investigated the necessary techniques that can be used and decided to apply Synthetic "
                "Minority Oversampling (SMOTE), an oversampling technique for unbalanced datasets. This technique is "
                "used to balance class distribution by creating synthetic samples.")

    st.header("Fine Tuning")
    st.markdown("After removing the imbalance in the data, we moved on to the training of the model. One of the most "
                "important things we learned in this step was to systematically search for the best combination of "
                "hyperparameters for our model.")

    st.markdown(
        "We understood the importance and concept of hyperparameters when we experienced that the performance of "
        "our model was significantly affected by the choice of hyperparameters. As teammates, we achieved "
        "different performances in our models and therefore different evaluation results, since the GridSearch "
        "algorithm found different values when we gave different parameter values.")

    st.header("Feature Importance")
    st.markdown(
        "When looking at the performance evaluations of our model, we checked the attribute importance order in our "
        "data and saw our newly created attribute 'crediworthness', which ranked high.")

    image = Image.open('website/feature_importance.png')
    st.image(image, caption='Feature Importance')

    st.subheader("Model")
    image = Image.open('website/xgboost.png')
    st.image(image, caption='XGBoost Ensemble Algorithm')