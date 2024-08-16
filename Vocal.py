import streamlit as st
st.title("vocal")
st.write("application streamlit")
import speech_recognition as sr
from pydub import AudioSegment
import io

# Fonction pour obtenir le texte à partir de l'audio
def transcribe_speech(audio_file, api_choice, language_code):
    recognizer = sr.Recognizer()
    try:
        # Charger le fichier audio
        audio = sr.AudioFile(audio_file)
        with audio as source:
            audio_data = recognizer.record(source)
        
        # Transcription en fonction de l'API choisie
        if api_choice == 'Google':
            text = recognizer.recognize_google(audio_data, language=language_code)
        elif api_choice == 'Sphinx':
            text = recognizer.recognize_sphinx(audio_data, language=language_code)
        else:
            raise ValueError("API de reconnaissance vocale non supportée.")
        
        return text, None
    except sr.UnknownValueError:
        return None, "Impossible de comprendre l'audio."
    except sr.RequestError as e:
        return None, f"Erreur de demande à l'API : {e}"
    except ValueError as ve:
        return None, str(ve)
    except Exception as e:
        return None, f"Erreur inattendue : {e}"

# Fonction pour sauvegarder le texte dans un fichier
def save_text_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

# Interface utilisateur Streamlit
st.title('Application de Reconnaissance Vocale')

# Choisir l'API de reconnaissance vocale
api_choice = st.selectbox('Choisissez l\'API de reconnaissance vocale', ['Google', 'Sphinx'])

# Choisir la langue
language_code = st.text_input('Code de langue (ex: en-US pour anglais, fr-FR pour français)', 'en-US')

# Télécharger le fichier audio
uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=['wav', 'mp3'])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # Transcription
    if st.button('Transcrire'):
        st.spinner('Transcription en cours...')
        text, error = transcribe_speech(uploaded_file, api_choice, language_code)
        
        if error:
            st.error(error)
        else:
            st.write('Texte transcrit :')
            st.write(text)

            # Sauvegarder le texte dans un fichier
            if st.button('Enregistrer le texte dans un fichier'):
                save_text_to_file(text, 'transcription.txt')
                st.success('Texte sauvegardé dans transcription.txt')

