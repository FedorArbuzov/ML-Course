#Select best metrics
#task2

# import numpy for loop
import numpy as np

# load dataset
from sklearn.datasets import load_boston

# for scaling parameters
from sklearn.preprocessing import scale

# for cross validation
from sklearn.model_selection import KFold

# for knn regression algorithm
from sklearn.neighbors import KNeighborsRegressor

# scaling data
boston = load_boston()
boston.data = scale(boston.data)

#create object for kNN
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# variable for best p in kNN
p_it = 0
p_val = -1

for p_main in np.linspace(1.0, 10.0, num=200):

    # arr for learning results
    arr_of_p = []

    # cross validation
    for train_indices, test_indices in kf.split(boston.data):
        # creating sets
        train_set_x = boston.data[train_indices]
        train_set_y = boston.target[train_indices]
        test_set_x = boston.data[test_indices]
        test_set_y = boston.target[test_indices]

        # kNN algorithm
        neigh = KNeighborsRegressor(n_neighbors=5, p=p_main, weights='distance')
        neigh.fit(train_set_x, train_set_y)

        # append learning results
        arr_of_p.append(neigh.score(test_set_x, test_set_y))

    # take average result
    temp_p_val = sum(arr_of_p) / len(arr_of_p)

    # if result is better change k
    if temp_p_val > p_val:
        p_val = temp_p_val
        p_it = p_main
print(p_it)