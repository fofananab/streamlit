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
# Vérification des valeurs manquantes
st.write("Valeurs manquantes par colonne :")
st.write(data.isnull().sum())

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

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import StandardScaler
#Standardiser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialiser le modèle
model = LinearRegression()

# Fonction pour effectuer la validation croisée
def evaluate_model(model, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    return -mse_scores, -mae_scores

# Interface utilisateur
st.title("Évaluation du Modèle de Prédiction des Locations de Vélo")
st.write("Ce modèle utilise la validation croisée pour évaluer ses performances.")

# Sélectionner le nombre de folds
folds = st.slider("Nombre de folds pour la validation croisée", min_value=2, max_value=10, value=5)

# Évaluer le modèle
if st.button("Évaluer le Modèle"):
    try:
        mse_scores, mae_scores = evaluate_model(model, X, y, folds)
        
        st.write(f"MSE moyen: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
        st.write(f"MAE moyen: {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")

        # Visualiser les résultats
        st.subheader("Scores de Validation Croisée")
        fig, ax = plt.subplots()
        ax.boxplot([mse_scores, mae_scores], labels=['MSE', 'MAE'])
        ax.set_title("Distribution des Scores de Validation Croisée")
        ax.set_ylabel("Erreur")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
        
    # Erreurs de prédiction
y_pred = model.fit(X, y).predict(X)
errors = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, errors, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Erreurs de Prédiction vs Valeurs Prédites')
plt.xlabel('Valeurs Prédites')
plt.ylabel('Erreurs')
plt.grid()
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, alpha=0.7, color='blue')
plt.title('Distribution des Erreurs de Prédiction')
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.grid()
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Ligne de référence
plt.title('Valeurs Réelles vs Valeurs Prédites')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Valeurs Prédites')
plt.grid()
st.pyplot(plt)

