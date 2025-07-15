import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load model dan label
model = load_model('face_emotion_vgg16.h5')
label_names = np.load('label_classes.npy')

st.title("Face Emotion Detection - CK+48 (VGG16)")
uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (48, 48))
    input_img = preprocess_input(np.expand_dims(img_resized, axis=0))

    pred = model.predict(input_img)
    pred_label = label_names[np.argmax(pred)]

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)
    st.write(f"### Predicted Emotion: {pred_label}")