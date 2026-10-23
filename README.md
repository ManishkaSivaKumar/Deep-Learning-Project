# Facial Emotion Recognition Using CNN (FER2013)

## 1. Project Overview
This project builds a beginner-friendly Convolutional Neural Network (CNN) using TensorFlow/Keras to classify facial emotions from grayscale face images.

Target classes (7):

1. Angry
2. Disgust
3. Fear
4. Happy
5. Sad
6. Surprise
7. Neutral

This repository is structured for a college CA1 deep learning assignment and is designed to run in Google Colab with GPU support.

## 2. Problem Statement
Given a 48x48 grayscale face image, predict the correct emotion class out of seven categories using a CNN model.

## 3. Dataset Source (FER2013)
Dataset format expected by this project:

- CSV file (usually `fer2013.csv`)
- Required columns:
  - `emotion` (integer labels 0 to 6)
  - `pixels` (space-separated 2304 pixel values)
- Optional column:
  - `Usage` (`Training`, `PublicTest`, `PrivateTest`)

How to obtain FER2013:

1. Official Kaggle challenge page (FER2013 dataset).
2. Public FER2013 mirrors in the same CSV format.

Important:
- Do not replace FER2013 with another dataset silently.
- If dataset download fails, fix download/access first and then run the notebook/scripts.

## 4. Project Architecture
Required CNN architecture implemented in `src/model.py`:

- Input: 48x48x1
- Conv2D(32, 3x3, ReLU)
- MaxPooling2D(2x2)
- Conv2D(64, 3x3, ReLU)
- MaxPooling2D(2x2)
- Conv2D(128, 3x3, ReLU)
- MaxPooling2D(2x2)
- Flatten
- Dense(128, ReLU)
- Dropout(0.5)
- Dense(7, Softmax)

Compile settings:

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy

## 5. Installation (Local)
```bash
pip install -r requirements.txt
```

## 6. Google Colab Execution Instructions (Recommended)

### Step A: Open notebook
1. Upload this project folder to Google Drive OR clone your repository in Colab.
2. Open `notebooks/facial_emotion_recognition.ipynb` in Colab.

### Step B: Enable GPU
1. Runtime -> Change runtime type
2. Hardware accelerator -> GPU

### Step C: Upload dataset
Use one of the following:

Option 1 (manual upload):
- Upload `fer2013.csv` to Colab runtime.
- Move it to `data/fer2013.csv`.

Option 2 (Google Drive):
- Mount Drive and set `CSV_PATH` to your file location.

### Step D: Run notebook cells in order
The notebook will:
1. Install/import libraries
2. Inspect and validate dataset columns
3. Parse/clean images
4. Split train/validation/test safely
5. Train CNN with callbacks
6. Evaluate on held-out test set
7. Plot curves and confusion matrix
8. Run image upload prediction demo
9. Save/load model

## 7. Training Instructions (Script)
From project root:

```bash
python src/train.py --csv_path data/fer2013.csv --epochs 20 --batch_size 64
```

Training features:

- EarlyStopping
- ModelCheckpoint (best model)
- ReduceLROnPlateau
- Optional augmentation (rotation, shift, horizontal flip)

Disable augmentation:

```bash
python src/train.py --csv_path data/fer2013.csv --no_augmentation
```

## 8. Evaluation Instructions

```bash
python src/evaluate.py --csv_path data/fer2013.csv --model_path models/best_emotion_model.keras
```

Generated outputs:

- Test loss and test accuracy
- Weighted precision, recall, F1-score
- Classification report
- Confusion matrix image

## 9. Prediction Demo Instructions

### In notebook (recommended)
Use the demo cell to upload an image and get:
- predicted emotion
- confidence score
- probabilities for all 7 classes

### Script usage
```bash
python src/predict.py --model_path models/best_emotion_model.keras --image_path your_image.jpg
```

## 10. Model Saving and Reuse
Model saving is automatic in training via callback:

- `models/best_emotion_model.keras`
- `models/class_mapping.json`

Load and predict later without retraining:

- `src/predict.py` loads saved model and class mapping.

To download from Colab:

```python
from google.colab import files
files.download('models/best_emotion_model.keras')
files.download('models/class_mapping.json')
```

## 11. Results Section (Placeholders)
Do not fill these before actual execution:

- Final Test Accuracy: `TO_BE_FILLED_AFTER_RUN`
- Final Test Loss: `TO_BE_FILLED_AFTER_RUN`
- Precision: `TO_BE_FILLED_AFTER_RUN`
- Recall: `TO_BE_FILLED_AFTER_RUN`
- F1-score: `TO_BE_FILLED_AFTER_RUN`

Add generated figures from:
- `results/plots/accuracy_curve.png`
- `results/plots/loss_curve.png`
- `results/metrics/confusion_matrix.png`

## 12. Limitations
- FER2013 images are low resolution (48x48), which limits detail.
- Emotion classes are somewhat subjective and can overlap.
- Class imbalance can affect minority class performance.
- Model is not a medical or psychological diagnosis system.

## 13. Future Improvements
- Class weighting or focal loss for imbalance.
- Better augmentation tuning.
- Hyperparameter search.
- Transfer learning baseline comparison.
- Real-time webcam inference extension.

## 14. Viva Preparation Tips
Be ready to explain:

1. Why normalization is needed.
2. Why train/val/test split avoids data leakage.
3. Why EarlyStopping and ModelCheckpoint are used.
4. Why confusion matrix and F1-score matter beyond accuracy.
5. Which classes are most confused and why.

## 15. Expected Project Structure

```text
facial-emotion-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── facial_emotion_recognition.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   └── README.md
│
├── reports/
│   └── project_report.md
│
└── demo/
    └── README.md
```
