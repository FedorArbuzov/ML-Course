#Normalization features
#task3

# numpy for converting dataframe to np.array
import numpy as np

# import perceptron
from sklearn.linear_model import Perceptron

# import function for accuracy
from sklearn.metrics import accuracy_score

# import pandas for csv reading
import pandas as pd

# import standard scale function
from sklearn.preprocessing import StandardScaler

# read csv table
df = pd.read_csv('perceptron-train.csv', header=None)

# get key value
Y = np.array(df[0])

# get other values
X = np.array(df[[1, 2]])

# read csv for testing
df_test = pd.read_csv('perceptron-test.csv', header=None)

# get key value
Y_test = np.array(df_test[0])

# get other values
X_test = np.array(df_test[[1, 2]])

# learning process
clf = Perceptron(random_state=241)
clf.fit(X, Y)

# create prediction for test set
predictions = clf.predict(X_test)

# accuracy result
print(accuracy_score(Y_test, predictions))

# scaler object
scaler = StandardScaler()

# get scale for test and train set
X_scaled = np.array(scaler.fit_transform(X))
X_test_scaled = np.array(scaler.transform(X_test))

# new learning process
clf1 = Perceptron(random_state=241)
clf1.fit(X_scaled, Y)

# create prediction for test set
new_predictions = clf.predict(X_test_scaled)

# accuracy result
print(accuracy_score(Y_test, new_predictions))

# result 0.56 ?
