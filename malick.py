import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
data=pd.read_csv('Expresso_churn.csv')
st.title("assata")
data.head()
data.info()
data.describe()

data.duplicated
data.drop_duplicates(inplace=True)

data.isnull().sum()
data.dropna(inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['TENURE']=le.fit_transform(data['TENURE'])
data['REGION']=le.fit_transform(data['REGION'])
data['TOP_PACK']=le.fit_transform(data['TOP_PACK'])
