import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    # Convertir en minuscules
    text = text.lower()
    # Tokenisation
    tokens = word_tokenize(text)
    # Retirer la ponctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Retirer les mots vides (stopwords)
    stop_words = set(stopwords.words('french'))  # Assurez-vous que les stopwords sont en français
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_relevant_sentence(user_query, sentences):
    # Prétraiter la requête de l'utilisateur
    user_query = preprocess(user_query)
    
    # Prétraiter les phrases du texte
    processed_sentences = [preprocess(sentence) for sentence in sentences]
    
    # Ajouter la requête de l'utilisateur à la liste des phrases
    processed_sentences.append(user_query)
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    # Calculer la similarité cosinus
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    # Trouver la phrase la plus similaire
    most_relevant_index = cosine_similarities.argmax()
    return sentences[most_relevant_index]

def chatbot(user_query, sentences):
    return get_most_relevant_sentence(user_query, sentences)

import streamlit as st

def load_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return sentences

def main():
    st.title("Chatbot sur les Moustiques")
    
    # Charger les phrases du fichier texte
    file_path = 'lesmoustiques.txt'  # Remplacez par le chemin vers votre fichier texte
    sentences = load_sentences(file_path)
    
    user_query = st.text_input("Posez une question sur les moustiques:")
    
    if user_query:
        response = chatbot(user_query, sentences)
        st.write("Réponse:", response)
    else:
        st.write("Entrez une question pour obtenir une réponse.")

if __name__ == "__main__":
    main()










