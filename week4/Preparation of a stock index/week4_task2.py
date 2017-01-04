# Preparation of a stock index
# task2


# for csv read
import pandas

# for converting df to np.array
import numpy as np

# for PCA algorithm
from sklearn.decomposition import PCA

# import correlation coefficient
from numpy import corrcoef

# read data
df = pandas.read_csv('close_prices.csv')

df = np.array(df.loc[:, 'AXP':])

# pca algorithm
pca = PCA(n_components=10)
pca.fit(df)

# find min number
v_glob = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    v_glob += v
    print(i, v_glob)
    if v_glob >= 0.9:
        print(i)
        break


# find correlation
df_comp = pandas.DataFrame(pca.transform(df))
comp0 = df_comp[0]


df2 = pandas.read_csv('djia_index.csv')
dji = df2['^DJI']
corr = corrcoef(comp0, dji)

print(corr[1, 0])

# max weight
df_new = pandas.read_csv('close_prices.csv')
df_new = df_new.loc[:, 'AXP':]
comp0_w = pandas.Series(pca.components_[0])
comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
print(comp0_w_top)
company = df_new.columns[comp0_w_top]
print(company)
