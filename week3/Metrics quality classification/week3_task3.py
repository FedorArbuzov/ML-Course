#Metrics quality classification
#task2

# numpy for converting dataframe to np.array
import numpy as np

# import pandas for csv reading
import pandas as pd

# import metrics from sklearn
from sklearn import metrics

# get data for 1 and 2 task
df = np.array(pd.read_csv('classification.csv'))


# Fill in the table of classification errors
TP = 0
FP = 0
FN = 0
TN = 0

for i in range(len(df)):
    #print(df[i, 0], df[i, 1])
    if df[i, 0] == df[i, 1] == 1:
        TP += 1
    elif df[i, 0] == df[i, 1] == 0:
        TN += 1
    elif df[i, 0] == 0 and df[i, 1] == 1:
        FP += 1
    else:
        FN += 1

print(TP, FP, FN, TN)


# Calculate the basic metric of quality of the classifier
print(metrics.accuracy_score(df[:, 0], df[:, 1]), metrics.precision_score(df[:, 0], df[:, 1]),
      metrics.recall_score(df[:, 0], df[:, 1]), metrics.f1_score(df[:, 0], df[:, 1]))

# get data for 3 and 4 task
df = np.array(pd.read_csv('scores.csv'))

# Calculate the area under the ROC-curve for each classifier. What qualifier has the highest metric value AUC-ROC
for i in range(4):
    print(metrics.roc_auc_score(df[:, 0], df[:, i + 1]))


# What classifier reaches the highest accuracy in the fullness of not less than 70%
k = 0
k_precision = 0
for i in range(4):
    precision, recall, thresholds = metrics.precision_recall_curve(df[:, 0], df[:, i + 1])
    for i1, val in enumerate(recall):
        if val > 0.7:
            if precision[i1] > k_precision:
                k = i + 1
                k_precision = precision[i1]
print(k)