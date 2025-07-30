import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load model dan label
model = load_model('face_emotion_vgg16.h5')
label_names = np.load('label_classes.npy')

st.title("Face Emotion Detection - CK+48 (VGG16)")
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Buka gambar dengan PIL dan ubah ke RGB
    img = Image.open(uploaded_file).convert('RGB')
    img_resized = img.resize((48, 48))  # Resize ke ukuran input model

    # Konversi ke array dan preprocess
    img_array = np.array(img_resized)
    input_img = preprocess_input(np.expand_dims(img_array, axis=0))

    # Prediksi
    pred = model.predict(input_img)
    pred_label = label_names[np.argmax(pred)]

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"### Predicted Emotion: {pred_label}")
