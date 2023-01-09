# libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as pyo

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,StandardScaler, RobustScaler

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
sns.set(rc={"figure.figsize":(12,12)})

data = pd.read_csv('data/BankChurners.csv')
data = data.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],
                axis='columns')


def create_afm(dataframe, csv=False):
    '''Creates customer segmentation with the AFM (Activity-Frequency-Monetary) score for a given data.

        Parameters
        ----------
        dataframe: dataframe
            The dataframe for calculating the AFM.
        csv: Boolean
            The output csv file of calculated AFM dataframe.

        Returns
        -------
            DataFrame

        Notes
        ------
        Requirements: pandas
        '''

    afm = pd.DataFrame()
    afm["client_id"] = data["CLIENTNUM"]
    afm["activity"] = (12 - data["Months_Inactive_12_mon"])
    afm["frequency"] = data["Contacts_Count_12_mon"]
    afm["monetary"] = data["Total_Trans_Amt"]

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

if __name__ == '__main__':
    afm = create_afm(data, True)
    print(afm)