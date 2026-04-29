#Iris classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler


data = pd.read_csv('Iris.csv')

print(data.head())

x= data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y= data['Species']

#Encode the target labels (Species)
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42,stratify=y, shuffle=True)

#Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Build Logistic Regression model
model= LogisticRegression( multi_class='multinomial', solver='lbfgs', max_iter=200)
print(model.fit(x_train, y_train))

y_pred = model.predict(x_test)
print(y_pred)

#accurecy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy", accuracy)

#Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot =True, fmt = 'd', cmap = 'Reds', xticklabels=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], yticklabels= ['Iris-setosa', 'Iris-versicolor','Iris-virginica'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#classification report
print(classification_report(y_test, y_pred, target_names=['Iris-setosa', 'Iris-versicolor','Iris-virginica']))

import joblib
joblib.dump(model, 'lrmodel_iris.pkl')
