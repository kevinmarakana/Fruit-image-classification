import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_data(folder, img_size=(100, 100), test_size=0.2):
    X, y = [], []
    classes = sorted(os.listdir(folder))
    for label in classes:
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path): continue
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label)
            except:
                continue
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    X = np.array(X, dtype=np.float32) / 255.0
    y_enc = np.array(y_enc)
    return train_test_split(X, y_enc, test_size=test_size, random_state=42), lb