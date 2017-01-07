# Gradient boosting of decision trees
# task 2

import math

import matplotlib.pyplot as plt

# import pandas for csv reading
import pandas as pd

# import numpy for np.array
import numpy as np

# import function for testing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# import log_loss from metrics
from sklearn.metrics import log_loss

# read data
gdm_data = np.array(pd.read_csv('gbm-data.csv'))

# get key value
y = gdm_data[:, 0]

# get other values
X = gdm_data[:, 1:]

# split for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def model_test(rate):

    sigmoid = lambda y_pred: 1.0 / (1.0 + math.exp(-y_pred))

    def log_loss_results(model, X, y):

        results = []
        for pred in model.staged_decision_function(X):
            results.append(log_loss(y, [sigmoid(y_pred) for y_pred in pred]))
        return results


    model = GradientBoostingClassifier(learning_rate=rate, n_estimators=250, verbose=True, random_state=241)
    model.fit(X_train, y_train)

    train_loss = log_loss_results(model, X_train, y_train)
    test_loss = log_loss_results(model, X_test, y_test)

    return plot_loss(rate, test_loss, train_loss)

def plot_loss(learning_rate, test_loss, train_loss):

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig('plots/rate_' + str(learning_rate) + '.png')
    min_loss_value = min(test_loss)
    min_loss_index = test_loss.index(min_loss_value)
    return min_loss_value, min_loss_index


min_loss_results = {}
learning_rate = [1, 0.5, 0.3, 0.2, 0.1]
for rate in learning_rate:
    min_loss_results[learning_rate] = model_test(rate)

min_loss_value, min_loss_index = min_loss_results[0.2]
print('{:0.2f} {}'.format(min_loss_value, min_loss_index))


model = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]
test_loss = log_loss(y_test, y_pred)
print(test_loss)
