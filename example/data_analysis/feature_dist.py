"""Plot correlation between features."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize

flen = 100
uncorr = False
correlation = 'pearson'

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=50, max_split=2000)
# Load data back in from save file.
ind = hv.load_split()

# Split out the various data.
train_data = np.array(hv._compile_split(ind['1_1'])[:, 1:-1], np.float64)
train_target = np.array(hv._compile_split(ind['1_1'])[:, -1:], np.float64)
test_data = np.array(hv._compile_split(ind['1_2'])[:, 1:-1], np.float64)
test_target = np.array(hv._compile_split(ind['1_2'])[:, -1:], np.float64)

# Scale and shape the data.
std = standardize(train_matrix=train_data, test_matrix=test_data)
train_data, test_data = std['train'], std['test']
train_target = train_target.reshape(len(train_target), )
test_target = test_target.reshape(len(test_target), )

corr = []
for c in train_data.T:
    if correlation is 'pearson':
        corr.append(pearsonr(c, train_target)[0])
    elif correlation is 'spearman':
        corr.append(spearmanr(c, train_target)[0])
    elif correlation is 'kendall':
        corr.append(kendalltau(c, train_target)[0])
ind = list(range(len(corr)))

sort_list = [list(i) for i in zip(*sorted(zip(ind, corr), key=lambda x: x[1],
                                          reverse=True))]

if uncorr:
    delf = sort_list[0][:len(ind)-flen]
else:
    delf = sort_list[0][flen:]
cut_ind = np.delete(ind, delf)
print(cut_ind)
train_data = np.delete(train_data, delf, axis=1)
print(np.shape(train_data))

traint = np.transpose(train_data)
ind = list(range(len(traint)))

d = {}
for i, j in zip(traint, ind):
    d[j] = i
df = pd.DataFrame(d)

ax = sns.violinplot(data=df, inner=None)
plt.show()
