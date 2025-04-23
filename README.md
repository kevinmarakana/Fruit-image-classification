# Fruit-image-classification"# Fruit-image-classification
 Dataset Link :  https://drive.google.com/drive/folders/1Rbuv5k0-xI5mFvTiorjVIHrbxEFsi1h2?usp=sharing

### Step Follow For Run Code

![result](https://github.com/user-attachments/assets/ab44369e-3b00-425c-abcc-50054d484058)

## Dataset
- `image_data/train/`: Training dataset with subfolders named by fruit type.
- `image_data/test/`: Testing dataset for final evaluation.
- `image_data/predict/`: Additional images for manual predictions.

## Features
- From-scratch CNN architecture
- Model training with validation split
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix and training history visualizations
- Prediction script for new unseen images
- Streamlit-based GUI for easy image upload and classification

## Running the Project
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run training:
```bash
python main.py
```
3. Launch GUI:
```bash
streamlit run gui/app.py
```

## Output Files
- `outputs/model.h5`: Best trained model
- `outputs/metrics.json`: Evaluation metrics
- `outputs/confusion_matrix.png`: Confusion matrix
- `outputs/training_plots.png`: Accuracy and loss over epochs

## Author
Kevin Marakana
