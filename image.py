import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Instructions pour l'utilisateur
st.title("Détection de Visages avec Streamlit")
st.write("""
    Cette application détecte les visages dans les images téléchargées.
    1. Téléchargez une image contenant des visages.
    2. Ajustez les paramètres de détection des visages comme `minNeighbors` et `scaleFactor`.
    3. Choisissez la couleur des rectangles autour des visages détectés.
    4. Cliquez sur le bouton pour voir l'image avec les visages détectés.
    5. Vous pouvez enregistrer l'image avec les visages détectés sur votre appareil.
""")

# Téléchargement de l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convertir l'image en niveaux de gris pour la détection des visages
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Charger le classificateur de visages
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Paramètres de détection des visages
    min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=3)
    scale_factor = st.slider("Scale Factor", min_value=1.01, max_value=2.0, step=0.01, value=1.1)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Choisir la couleur du rectangle
    color = st.color_picker("Choisissez la couleur du rectangle", "#FF0000")
    color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

    # Dessiner les rectangles autour des visages
    for (x, y, w, h) in faces:
        cv2.rectangle(image_np, (x, y), (x+w, y+h), color, 2)

    # Convertir l'image en format utilisable pour Streamlit
    image_with_faces = Image.fromarray(image_np)

    # Afficher l'image avec les visages détectés
    st.image(image_with_faces, caption="Image avec Visages Détectés")

    # Enregistrer l'image
    buffer = io.BytesIO()
    image_with_faces.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Télécharger l'image avec visages détectés",
        data=buffer,
        file_name="image_avec_visages_detectes.png",
        mime="image/png"
    )


