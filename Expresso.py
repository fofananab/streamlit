import pandas as pd
import streamlit as st

st.title("Pandas")
data=pd.read_csv('Expresso_churn.csv')

data.describe()
data.isnull().sum()
data.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['TENURE']=le.fit_transform(data['TENURE'])
data['REGION']=le.fit_transform(data['REGION'])
data['TOP_PACK']=le.fit_transform(data['TOP_PACK'])
data['MRG']=le.fit_transform(data['MRG'])
data['ORANGE']=le.fit_transform(data['MRG'])
data['TIGO']=le.fit_transform(data['MRG'])
data['ON_NET']=le.fit_transform(data['MRG'])
# Example: Applying the same method to multiple columns
value = 10
data[['col1', 'col2']] = data[['MONTANT', 'REVENUE']].apply(lambda x: x)



data.head()
st.dataframe(data)
import  plotly.express  as  px 
st.title("analyse ")
# Ce dataframe comporte 244 lignes, mais 4 valeurs distinctes pour `day`
fig  =  px . pie (data ,     names = 'REGION' ) 
fig . show ()
event = st.plotly_chart(fig, key="data", on_select="rerun")
event

 # Ou utiliser pd.get_dummies pour des variables catégorielles avec plusieurs niveaux
df_encoded = pd.get_dummies(data)
df_encoded.head()

st.title("orange")

fig = px.bar(data, x="REGION", y="MONTANT", color="ORANGE", barmode="group")

st.plotly_chart(fig)

st.title("TIGO")
fig = px.bar(data, x="REGION", y="MONTANT", color="TIGO", barmode="group")

st.plotly_chart(fig)

import plotly.graph_objects as go

fig = go.Figure(go.Waterfall(
    name = "20", orientation = "v",
    measure = ["relative", "relative", "total", "relative", "relative", "total"],
    x = ["MRG", "REVENUE", "ORANGE", "TIGO", "ON_NET", "MONTANT"],
    textposition = "outside",
    text = ["+60", "+80", "", "-40", "-20", "Total"],
    y = [60, 80, 0, -40, -20, 0],
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
))

fig.update_layout(
        title = "A_CHURN",
        showlegend = True
)

fig.show()
st.plotly_chart(fig)
st.sidebar.success("visualisation")
st.title("prediction")
from sklearn.model_selection import train_test_split


# Supposons X contient toutes les colonnes sauf la colonne cible et y est la colonne cible
X = data.drop(columns=['CHURN'])
y = data['CHURN']
X
y

# Fractionnement en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=1/4, random_state=0)

# 'test_size=0.2' spécifie que 20% des données sont utilisées comme ensemble de test.
# 'random_state=42' garantit que la division est reproductible.

# Afficher les dimensions des ensembles résultants
print("Taille de l'ensemble d'entraînement:", X_train.shape, y_train.shape)
print("Taille de l'ensemble de test:", X_test.shape, y_test.shape)




st.sidebar.success("Machine learning")

import joblib
from sklearn.ensemble import RandomForestClassifier

# Entraînez un modèle
model = RandomForestClassifier()
# Entraînez-le avec vos données ici...

# Sauvegardez le modèle
joblib.dump(model, 'model.pkl')

import streamlit as st
import joblib
import numpy as np

# Chargez le modèle pré-entraîné
model = joblib.load('model.pkl')

# Créez l'interface utilisateur avec des champs de saisie
st.title('Prédiction de Machine Learning')

# Remplacez les champs ci-dessous par les fonctionnalités spécifiques de votre modèle
feature1 = st.number_input('CHURN', min_value=0.0, max_value=100.0, value=50.0)
feature2 = st.number_input('MONTANT', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.number_input('REVENUE', min_value=0.0, max_value=100.0, value=50.0)
feature4 = st.number_input('ORANGE', min_value=0.0, max_value=100.0, value=50.0)

# Convertir les entrées utilisateur en tableau numpy
user_input = np.array([['MONTANT', 'REVENUE', 'CHURN', 'ORANGE']])

# Ajouter un bouton pour faire la prédiction
if st.button('Faire une prédiction'):
    prediction = model.predict(user_input)
    st.write(f'Prédiction : {prediction[0]}')

st.sidebar.success("STRIMLIT")
