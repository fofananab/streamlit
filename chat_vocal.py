import nltk
import streamlit as st
import speech_recognition as sr
from nltk.chat.util import Chat, reflections

# Exemple de données de conversation
chatbot_pairs = [
    (r"Bonjour", "Bonjour! Comment puis-je vous aider aujourd'hui?"),
    (r"(.*) votre nom (.*)", "Je suis un chatbot créé pour vous aider."),
    # Ajoutez plus de paires question-réponse ici
]

# Créez un chatbot à l'aide de NLTK
chatbot = Chat(chatbot_pairs, reflections)

def transcrire_parole_en_texte():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Veuillez parler...")
        audio = recognizer.listen(source)
        try:
            texte = recognizer.recognize_google(audio, language="fr-FR")
            st.write(f"Vous avez dit : {texte}")
            return texte
        except sr.UnknownValueError:
            st.write("Je n'ai pas pu comprendre l'audio.")
            return None
        except sr.RequestError:
            st.write("Erreur de connexion au service de reconnaissance vocale.")
            return None
def obtenir_reponse_utilisateur(texte_utilisateur):
    if texte_utilisateur:
        reponse = chatbot.respond(texte_utilisateur)
        return reponse
    return "Je n'ai pas compris votre message."
def app():
    st.title("Chatbot à Commande Vocale")
    
    if st.button("Parlez maintenant"):
        texte = transcrire_parole_en_texte()
        if texte:
            reponse = obtenir_reponse_utilisateur(texte)
            st.write("Chatbot:", reponse)

# Exécuter l'application Streamlit
if __name__ == "__main__":
    app()

