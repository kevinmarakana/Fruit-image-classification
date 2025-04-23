import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
    
def evaluate_model(model, X_test, y_test, lb):
    y_pred = model.predict(X_test)
    y_pred_labels = lb.inverse_transform(y_pred)
    y_true_labels = lb.inverse_transform(y_test)

    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
    with open('outputs/metrics.json', 'w') as f:
        json.dump(report, f, indent=4)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=lb.classes_)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('outputs/confusion_matrix.png')