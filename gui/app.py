import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.predict import predict_image

st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title("🍎 Fruit Image Classifier")

model = load_model("outputs/model.h5")
(X_train, X_val, y_train, y_val), lb = load_data('image_data/train')

uploaded_file = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(img, (100, 100)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_input)
    pred_label = lb.classes_[np.argmax(prediction)]

    st.image(img, channels="BGR", caption="Uploaded Image", width=300)
    st.subheader(f"🔍 Predicted Fruit: {pred_label}")