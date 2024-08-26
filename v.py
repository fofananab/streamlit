import nltk
import streamlit as st
import speech_recognition as sr
from nltk.chat.util import Chat, reflections
# Définir les paires de réponses du chatbot
pairs = [
    (r'hi|hello', ['Hello! How can I help you today?']),
    (r'what is your name?', ['I am a chatbot created to assist you.']),
    (r'how are you?', ['I am just a bot, but I am doing well!']),
    (r'(.*)', ['I am not sure how to respond to that.'])
]

# Créer l'instance du chatbot
chatbot = Chat(pairs, reflections)

def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            st.write("Sorry, there was a problem with the speech recognition service.")
            return ""
def get_chatbot_response(user_input):
    return chatbot.respond(user_input)
def main():
    st.title("Chatbot with Speech Recognition")

    option = st.radio("Choose input method:", ("Text", "Voice"))

    if option == "Text":
        user_input = st.text_input("You: ", "")
        if user_input:
            response = get_chatbot_response(user_input)
            st.write(f"Chatbot: {response}")

    elif option == "Voice":
        if st.button("Start Recording"):
            user_input = transcribe_speech()
            if user_input:
                response = get_chatbot_response(user_input)
                st.write(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
