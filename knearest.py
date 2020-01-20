import pandas as pd
import numpy as np
import math as mt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
train=pd.read_csv("train-data.txt", delimiter=' ',engine='python',header=None)
print(train)
#print(train.columns)
#print(train[1])
print(train.head())
test=pd.read_csv("test-data.txt", delimiter=' ',engine='python',header=None)
print(test)
train_input=train.iloc[:,2:194].values
train_label= train.iloc[:,1].values
test_input=test.iloc[:,2:194].values
test_label=test.iloc[:,1].values
classifier = KNeighborsClassifier(n_neighbors=int(input('enter the k value')))
classifier.fit(train_input, train_label)
y_pred = classifier.predict(test_input)
print(confusion_matrix(test_label, y_pred))
print(classification_report(test_label, y_pred))