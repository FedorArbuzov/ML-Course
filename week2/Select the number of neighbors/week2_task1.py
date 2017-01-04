#Select the number of neighbors
#task1

# for csv read
import pandas

# for knn algorithm 
from sklearn.neighbors import KNeighborsClassifier

# for cross validation
from sklearn.model_selection import KFold

# for scaling parameters
from sklearn.preprocessing import scale


# read data
df = pandas.read_csv('wine.data', header=None,)
y = df[0]

# scaling data
x = scale(df[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]])

#create object for kNN
kf = KFold(n_splits=5, random_state=42, shuffle=True)

# variable for best k in kNN
k_it = 0
k_val = -1

# loop for finding best k
for k in range(1, 51):

    # arr for learning results
    arr_of_k = []

    #cross validation
    for train_indices, test_indices in kf.split(x):

        #creating sets
        train_set_x = x[train_indices]
        train_set_y = y[train_indices]
        test_set_x = x[test_indices]
        test_set_y = y[test_indices]

        #kNN algorithm
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_set_x, train_set_y)

        #append learning results
        arr_of_k.append(neigh.score(test_set_x, test_set_y))

    # take average result
    temp_k_val = sum(arr_of_k)/len(arr_of_k)

    # if result is better change k
    if temp_k_val > k_val:
        k_val = temp_k_val
        k_it = k


print(k_it, k_val)
# result: 1 0.730476190476 29 0.977619047619
