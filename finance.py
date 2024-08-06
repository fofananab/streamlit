from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
st.title("ANALYSE :bar_chart::coffee:")
df=pd.read_csv('Financial.csv')
st.dataframe(df)
df.head(5)

df.info()

df.describe()

df.isnull().sum()

df.shape

df.drop_duplicates()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df['uniqueid']=le.fit_transform(df['uniqueid'])
df['bank_account']=le.fit_transform(df['bank_account'])
df['location_type']=le.fit_transform(df['location_type'])
df['cellphone_access']=le.fit_transform(df['cellphone_access'])
df['gender_of_respondent']=le.fit_transform(df['gender_of_respondent'])
df['relationship_with_head']=le.fit_transform(df['relationship_with_head'])
df['marital_status']=le.fit_transform(df['marital_status'])
df['education_level']=le.fit_transform(df['education_level'])
df['job_type']=le.fit_transform(df['job_type'])
df['year']=le.fit_transform(df['year'])
df['household_size']=le.fit_transform(df['household_size'])
df.head(5)
st.dataframe(df)


mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

import plotly.express as px


chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["age_of_respondent", "relationship_with_head", "cellphone_access"])

st.bar_chart(chart_data)


st.bar_chart(df, x="age_of_respondent", y="education_level", color="country", horizontal=True)




st.bar_chart(df, x="year", y="age_of_respondent", color="job_type", stack=False)

# the histogram of the data


st.title("prediction")
from sklearn.model_selection import train_test_split
x = df[['age_of_respondent', 'gender_of_respondent','job_type']]
y = df['relationship_with_head']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0) 
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
model = LogisticRegression(max_iter=200)

pipeline = make_pipeline(scaler, model)
pipeline.fit(x_train, y_train)  # X_train et y_train sont vos données d'entraînement



logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.1f}".format(logreg.score(x_test, y_test)))

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
figy=sns.regplot(x='age_of_respondent',y='job_type',data=df)
ax.set_title('Graphique de Dispersion')

# Titre de l'application
st.title('Visualisation avec Seaborn et Streamlit')



# Afficher la figure dans Streamlit
st.pyplot(fig)  # Assurez-vous que 'fig' est une 

import matplotlib.pyplot as plt

cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# Fonction pour afficher la matrice de confusion
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    ax.set_xlabel('Prédictions')
    ax.set_ylabel('Véritables')
    ax.set_title('Matrice de Confusion')
    st.pyplot(fig)

# Titre de l'application
st.title('Visualisation de la Matrice de Confusion avec Streamlit et Seaborn')

# Afficher la matrice de confusion
plot_confusion_matrix(cm)

st.sidebar.success("pandas")

from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print(classification_report(y_test,y_pred))
st.text(classification_report(y_test,y_pred))
st.sidebar.success("Machine_learning")

# Titre de l'application
st.title('man')

# Titre de l'application
st.title('Prédiction avec modèle ML')

# Ajout des champs de saisie pour les caractéristiques
st.header('Entrez les valeurs des caractéristiques')

# Exemple de champs de saisie pour les caractéristiques (modifiables selon vos données)
age = st.number_input('Âge', min_value=18, max_value=100, value=25)
income = st.number_input('Revenu annuel', min_value=0, max_value=200000, value=50000)
education = st.selectbox('Niveau d\'éducation', ['Bachelier', 'Master', 'Doctorat'])

# Transformer les valeurs catégorielles en variables indicatrices
education_encoded = 0
if education == 'Master':
    education_encoded = 1
elif education == 'Doctorat':
    education_encoded = 2

# Créer un dataframe avec les valeurs saisies
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Education': [education_encoded]
})



# Faire des prédictions avec le modèle chargé
if st.button('Faire la prédiction'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Afficher la prédiction et la probabilité de chaque classe
    st.header('Résultats de la prédiction')
    st.write(f'Prédiction : {prediction[0]}')
    st.write(f'Probabilités : {prediction_proba}')

