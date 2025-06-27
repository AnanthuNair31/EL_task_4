# EL_task_4

## Objective
The goal of this task was to build a binary classification model using Logistic Regression to accurately classify breast cancer tumors as either Malignant (M) or Benign (B) using a dataset of medical features.
## Tools & Libraries Used
Python
pandas for data handling
scikit-learn for modeling and evaluation
matplotlib for visualization
## Dataset
Total Samples: 569
Target Variable: diagnosis (M for malignant, B for benign)
Features: 30 real-valued input features related to tumor characteristics such as radius, texture, perimeter, etc.
## Data Preprocessing
Converted the categorical diagnosis column into binary numeric values (M → 1, B → 0)
Removed irrelevant columns (id, empty column Unnamed: 32)
Checked for null values and data types
## Feature Scaling
Standardized features using StandardScaler to normalize value ranges
## Model Training
Trained a Logistic Regression model on the training set
## Model Evaluation
Evaluated the model using:
Confusion Matrix
Classification Report (Precision, Recall)
ROC-AUC Score
Achieved high accuracy (~97%) with ROC-AUC ≈ 0.997
## Threshold Tuning and Sigmoid Function
Explained the sigmoid function used in logistic regression:
Tuned the decision threshold from 0.0 to 1.0 in steps to analyze how precision and recall vary
Plotted precision-recall vs threshold and identified an optimal threshold
Handled undefined precision values using zero_division=0

