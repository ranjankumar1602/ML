"""Importing the dependencies used"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection and Data Processing"""

#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('sonar data.csv', header=None)

sonar_data.head()

# number of rows and columns
sonar_data.shape

sonar_data.describe()  #describe --> statistical measures of the data

sonar_data[60].value_counts()

"""M --> Mine

R --> Rock
"""

sonar_data.groupby(60).mean()

# separating data and Labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)

"""Training and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

print(X_train)
print(Y_train)

"""Model Training --> Logistic Regression"""

model = LogisticRegression()

#training the Logistic Regression model with training data
model.fit(X_train, Y_train)

"""Model Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ', test_data_accuracy)

"""Making a Predictive System"""

input_data = (0.0530,0.0885,0.1997,0.2604,0.3225,0.2247,0.0617,0.2287,0.0950,0.0740,0.1610,0.2226,0.2703,0.3365,0.4266,0.4144,0.5655,0.6921,0.8547,0.9234,0.9171,1.0000,0.9532,0.9101,0.8337,0.7053,0.6534,0.4483,0.2460,0.2020,0.1446,0.0994,0.1510,0.2392,0.4434,0.5023,0.4441,0.4571,0.3927,0.2900,0.3408,0.4990,0.3632,0.1387,0.1800,0.1299,0.0523,0.0817,0.0469,0.0114,0.0299,0.0244,0.0199,0.0257,0.0082,0.0151,0.0171,0.0146,0.0134,0.0056)

# changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a mine')