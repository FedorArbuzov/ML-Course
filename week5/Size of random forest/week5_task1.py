#Size of random forest
#task1

# numpy for converting dataframe to np.array
import numpy as np

# import pandas for csv reading
import pandas as pd

# import Random forest regressor
from sklearn.ensemble import RandomForestRegressor

# for cross validation
from sklearn.model_selection import KFold

from sklearn.metrics import r2_score
# read csv table
df = np.array(pd.read_csv('abalone.csv'))

# get key value
y = df[:, -1]

# get other values
X = df[:, 0:8]
for i in X:
    if i[0] == 'M':
        i[0] = 1
    elif i[0] == 'F':
        i[0] = -1
    else:
        i[0] = 0


#create object for cross validation
kf = KFold(n_splits=5, random_state=1, shuffle=True)

for i in range(1, 51):
    print(i)
    # arr for learning results
    arr_of_k = []

    # cross validation
    for train_indices, test_indices in kf.split(X):

        # creating sets
        train_set_x = X[train_indices]
        train_set_y = y[train_indices]
        test_set_x = X[test_indices]
        test_set_y = y[test_indices]

        # learning algorithm
        clf = RandomForestRegressor(n_estimators=i, random_state=1)
        clf.fit(train_set_x, train_set_y)

        # append learning results
        arr_of_k.append(r2_score(test_set_y, clf.predict(test_set_x)))

    # take average result
    temp_k_val = sum(arr_of_k) / len(arr_of_k)
    # if result is better
    if temp_k_val > 0.52:
        print(i)
        break
