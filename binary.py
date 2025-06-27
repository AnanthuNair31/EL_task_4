import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt


df= pd.read_csv("data.csv")

df_info = df.info()
df_head = df.head()
df_class_distribution = df.iloc[:, -1].value_counts()
print(df_info , df_head , df_class_distribution)

print(df['Unnamed: 32'].head())

df_clean = df.drop(columns=['Unnamed: 32', 'id'])

df_clean['diagnosis'] = df_clean['diagnosis'].map({'M': 1 , 'B' : 0})

x = df_clean.drop( columns = 'diagnosis')
y = df_clean['diagnosis']

x_train , x_test, y_train , y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()

x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)

print(x_train_sc.shape, x_test_sc.shape, y_train.value_counts() , y_test.value_counts())

log_reg = LogisticRegression(random_state=42 , max_iter=1000)
log_reg.fit(x_train_sc, y_train)

y_pred = log_reg.predict(x_test_sc)
y_proba = log_reg.predict_proba(x_test_sc)[: , 1]

print("Model Coefficients:\n", log_reg.coef_)
print("Intercept:", log_reg.intercept_)

cf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cf_matrix)

print('Classification report:\n', classification_report(y_test, y_pred , target_names =['Bengin' , 'Malignant']))

roc_auc = roc_auc_score(y_test, y_proba)
print('ruc-aoc score:', roc_auc)

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title('ROC Curve')
plt.grid(True)
plt.show()

thresholds = np.arange(0, 1.05, 0.05)
precisions = []
recalls = []

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))

plt.figure(figsize=(12, 8))
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall' , marker='s' )
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall Curve vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

#SIGMOID FUNCTION
'''The sigmoid output gives the probability that the sample belongs to the positive class (in our case: Malignant, i.e., 1).
y_proba stores the sigmoid probabilities that each tumor is malignant (class 1).These values lie between 0 and 1 (due to the sigmoid)'''

