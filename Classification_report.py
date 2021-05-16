# Decision Tree Classifier

# Importing required Libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Importing the Dataset from my GitHub Repository
url = "https://raw.githubusercontent.com/tanishkthomas/Datasets/main/Iris.csv"
dataset = pd.read_csv(url)
X = dataset.iloc[:, 1:5]
Y = dataset.iloc[:, 5]



encoder = LabelEncoder()
Y = encoder.fit_transform(Y)


# Splitting of data into Train and Test sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)


# Scaling the Data to avoid bias and normalising it to a particular range
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)


# Fitting the classifier
classifier = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, random_state=42)
classifier.fit(X_train, Y_train)


# Predicting the test values
Y_pred = classifier.predict(X_test)


# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
'''array([[19,  0,  0],
          [ 0, 11,  2],
          [ 0,  0, 13]], dtype=int64)'''  # 70:30

'''array([[10,  0,  0],
          [ 0,  9,  0],
          [ 0,  0, 11]], dtype=int64)'''  # 80:20

'''array([[23,  0,  0],
          [ 0, 19,  0],
          [ 0,  1, 17]], dtype=int64)'''  # 60:40

# Getting the number of unique values in the Y_test set, in order to compare it with the Confusion Matrix
# np.unique(Y_test, return_counts=True)
'''(array([0, 1, 2]), array([19, 13, 13], dtype=int64))'''  # 70:30
'''(array([0, 1, 2]), array([10,  9, 11], dtype=int64))'''  # 80:20
'''(array([0, 1, 2]), array([23, 19, 18], dtype=int64))'''  # 60:40


# 60:40
# precision_score(Y_test, Y_pred, average='weighted')
# Out[19]: 0.9841666666666666
# recall_score(Y_test, Y_pred, average='weighted')
# Out[20]: 0.9833333333333333
# f1_score(Y_test, Y_pred, average='weighted')
# Out[21]: 0.9833089133089132
# accuracy_score(Y_test, Y_pred)
# Out[23]: 0.9833333333333333
cr6040 = classification_report(Y_test, Y_pred)

fpr, tpr, thresh = roc_curve(Y_test, Y_pred)

