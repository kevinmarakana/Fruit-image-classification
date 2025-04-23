import numpy as np
from tensorflow.keras.models import load_model
import cv2

def predict_image(img_path, lb, img_size=(100, 100)):
    model = load_model('outputs/model.h5')
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return lb.classes_[np.argmax(prediction)]