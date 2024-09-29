import pandas as pd
import numpy as np
from io import StringIO
import streamlit as st
import matplotlib.pyplot  as plt
st.title("it is my projet")
st.title("pandas")

data=pd.read_csv("day.csv")
print(data.dtypes)
data.head()
data.info()
data.describe()
data.isnull()
st.dataframe(data)

st.sidebar.success("ANALYSE")
import matplotlib.pyplot as plt

data[['temp', 'atemp', 'hum', 'windspeed']].hist(bins=20, figsize=(10, 7))

st.pyplot(plt)

data.plot.scatter(x='temp', y='cnt', alpha=0.5)
plt.title('Température vs Nombre total de locations')
st.pyplot(plt)

import  plotly.express  as  px
 
fig  =  px . bar ( data,  x = 'mnth' ,  y = 'cnt' )
 
st.plotly_chart(fig, use_container_width=True)


fig  =  px.bar(data,  x='weekday',  y='cnt',
               
hover_data=['temp', 'hum'],  color='temp') # Changed hover_data to valid column names 'temp' and 'h

st.plotly_chart(fig, use_container_width=True)


st.sidebar.success("visualisation")

import matplotlib.pyplot  as plt
from scipy import stats
#Normal plot
fig=plt.figure(figsize=(15,8))
stats.probplot(data.cnt.tolist(),dist='norm',plot=plt)
st.pyplot(fig)
 #Convertir la colonne 'date' en datetime
data['dteday'] = pd.to_datetime(data['dteday'])

corr=data.corr()
import seaborn as sns

plt.figure(figsize=(20,20))
sns.heatmap(corr,cmap="crest",annot=True)
st.pyplot(plt)
st.sidebar.success("CORRELATION")

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Sélection des caractéristiques
features = data[['instant', 'cnt', 'temp']]

# Déterminer le nombre optimal de clusters
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)

# Visualiser la méthode du coude
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Nombre de clusters K')
plt.ylabel('Inertie')
plt.title('Méthode du coude')
plt.show()
st.pyplot(plt)

# Choisir K (par exemple K=3)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal)
data['cluster'] = kmeans.fit_predict(features)

# Ajouter les clusters aux caractéristiques
X = data[['instant', 'atemp', 'temp', 'cluster']]

y = data['cnt']  # Variable cible

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
y_pred

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Erreur Quadratique Moyenne: {mse}')
print(f'Score R²: {r2}')

# Visualiser les prédictions par rapport aux valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Valeurs réelles')
plt.ylabel('Prédictions')
plt.title('Prédictions vs Valeurs Réelles')
plt.show()
st.pyplot(plt)

st.sidebar.success("PREDICTION")

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

st. title("Liste des modèles")
models={
    "Régression Linéaire": LinearRegression(),
    "Arbre de Décision": DecisionTreeRegressor(),
    "Forêt Aléatoire": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
}
st.write(models)

# Évaluation par validation croisée
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    results[name] = -scores.mean()  # Prendre la moyenne de l'erreur
# Affichage des résultats
for name, score in results.items():
 
 st.write(f"{name}: Erreur Quadratique Moyenne = {score:.2f}")
st.sidebar.success("les autres models")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Prétraitement des données
data['dteday'] = pd.to_datetime(data['dteday'])
data['day'] = data['dteday'].dt.day
data['month'] = data['dteday'].dt.month
data['year'] = data['dteday'].dt.year

# Sélection des caractéristiques
features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
            'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

X = data[features]
y = data['cnt']  # Variable cible

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline pour le modèle
model = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', 'passthrough', ['temp', 'atemp', 'hum', 'windspeed']),
        ('cat', OneHotEncoder(), ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'])
    ])),
    ('regressor', LinearRegression())
])

# Entraîner le modèle
model.fit(X_train, y_train)

# Streamlit UI
st.title("model de regression")

# User input for predictions
st.header("Enter Features for Prediction")

season = st.selectbox("Season", options=[1, 2, 3, 4])
yr = st.selectbox("Year (0: 2011, 1: 2012)", options=[0, 1])
mnth = st.selectbox("Month", options=list(range(1, 13)))
holiday = st.selectbox("Holiday (0: No, 1: Yes)", options=[0, 1])
weekday = st.selectbox("Weekday (0: Sunday, 6: Saturday)", options=list(range(0, 7)))
workingday = st.selectbox("Working Day (0: No, 1: Yes)", options=[0, 1])
weathersit = st.selectbox("Weather Situation (1 to 3)", options=[1, 2, 3])
temp = st.number_input("Temperature (Celsius)", min_value=0.0, max_value=40.0, value=20.0)
atemp = st.number_input("Feeling Temperature (Celsius)", min_value=0.0, max_value=40.0, value=20.0)
hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
windspeed = st.number_input("Windspeed (m/s)", min_value=0.0, max_value=20.0, value=5.0)

# Prepare the input for prediction
input_data = pd.DataFrame({
    'season': [season],
    'yr': [yr],
    'mnth': [mnth],
    'holiday': [holiday],
    'weekday': [weekday],
    'workingday': [workingday],
    'weathersit': [weathersit],
    'temp': [temp],
    'atemp': [atemp],
    'hum': [hum],
    'windspeed': [windspeed]
})

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted Bike Rentals: {int(prediction[0])}")
st.sidebar.success("model de regression")
