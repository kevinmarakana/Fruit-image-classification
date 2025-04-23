from src.data_loader import load_data
from src.model_builder import build_model
from src.train_model import train_model
from src.evaluate_model import evaluate_model

# Main script to load data, build model, train, and evaluate

(X_train, X_val, y_train, y_val), lb = load_data('image_data/train')
model = build_model((100, 100, 3), len(lb.classes_))
train_model(model, X_train, y_train, X_val, y_val)
evaluate_model(model, X_val, y_val, lb)