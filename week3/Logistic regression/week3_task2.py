#Logistic regression
#task2

# numpy for converting dataframe to np.array
import numpy as np

# import pandas for csv reading
import pandas as pd

# read csv table
df = pd.read_csv('data-logistic.csv', header=None)

# get key value
y = np.array(df[0])

# get other values
X = np.array(df[[1, 2]])

print(y[0])
print(X[0])