import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
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

x = data[['CHURN', 'REGION','ON_NET']]
y = data['MONTANT']

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
figy=sns.regplot(x='CHURN',y='MONTANT',data=data)
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



from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

print(classification_report(y_test,y_pred))
st.text(classification_report(y_test,y_pred))
st.sidebar.success("Machine_learning")

# Titre de l'application
st.title('man')

# Ajout des champs de saisie
name = st.text_input('Entrez votre nom', 'John Doe')
age = st.number_input('Entrez votre âge', min_value=0, max_value=130, value=0)

# Affichage des résultats
st.write('Nom :', name)
st.write('Âge :', age)
# Titre de l'application
st.title('man')
# Titre de l'application
st.title('Application de Classification')

# Section pour uploader un fichier de données
st.sidebar.header('Uploader le fichier de données')
uploaded_file = st.sidebar.file_uploader("Expresso_churn_dataset (2) CSV", type=["csv"])

# Si un fichier est chargé, affiche les premières lignes du fichier
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

# Section pour sélectionner le modèle et lancer l'entraînement
st.sidebar.header('Paramètres du modèle')
classifier_name = st.sidebar.selectbox(
    'Sélectionner le modèle de classification',
    ('Régression Logistique', 'SVM', 'Random Forest')
)

# Sélection du modèle
if classifier_name == 'Régression Logistique':
    model = LogisticRegression()
elif classifier_name == 'SVM':
    model = SVC()
elif classifier_name == 'Random Forest':
    model = RandomForestClassifier()

# Affichage des paramètres du modèle
st.write('Paramètres du modèle sélectionné:', model.get_params())

# Section pour l'entraînement et l'évaluation du modèle
if st.button('Entraîner le modèle'):
    # Ici, vous pouvez ajouter le code pour entraîner votre modèle avec les données chargées
    # et afficher les résultats d'évaluation

    # Indent the following line to fix the error
    st.sidebar.success('Modèle entraîné avec succès!')

#Rest of the code remains unchanged
