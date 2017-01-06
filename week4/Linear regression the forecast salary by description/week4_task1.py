# Linear regression the forecast salary by description
# task 1


# import pandas for read_csv
import pandas as pd

# one-hot-coding
from sklearn.feature_extraction import DictVectorizer

# linear regression
from sklearn.linear_model import Ridge

# for matrix creating
from scipy.sparse import hstack

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

# text pretreatment
def text_pretreatment(text):

    # get lower case
    text = text.map(lambda x: x.lower())

    # remove all except letters
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)

    return text


# read csv
salary_train = pd.read_csv('salary-train.csv')

# tf-idf algorithm
vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_pretreatment(salary_train['FullDescription']))

# remove nan elements
salary_train['LocationNormalized'].fillna('nan', inplace=True)
salary_train['ContractTime'].fillna('nan', inplace=True)


enc = DictVectorizer()
X_train_cat = enc.fit_transform(salary_train[['LocationNormalized', 'ContractTime']].to_dict('records'))


X_train = hstack([X_train_text, X_train_cat])

# logistic regression
y_train = salary_train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)

# read test csv
test = pd.read_csv('salary-test-mini.csv')

# test algorithm
X_test_text = vec.transform(text_pretreatment(test['FullDescription']))
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)
print('{:0.2f} {:0.2f}'.format(y_test[0], y_test[1]))
