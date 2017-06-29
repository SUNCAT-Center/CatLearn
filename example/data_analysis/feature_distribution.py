"""Plot feature distribution sorted by correlation with target."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize
from atoml.utilities import clean_variance

flen = 100  # Number of features to check.
cut = True  # True for best correlated, False for least.
correlation = 'pearson'  # Type of correlation analysis.

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
train_data = clean_variance(train_data)['train']

# Scale and shape the data.
train_data = standardize(train_matrix=train_data)['train']
d, f = np.shape(train_data)
train_target = train_target.reshape(len(train_target), )
traint = np.transpose(train_data)
index = list(range(f))

corr = []
for c in traint:
    if correlation is 'pearson':
        corr.append(pearsonr(c, train_target)[0])
    elif correlation is 'spearman':
        corr.append(spearmanr(c, train_target)[0])
    elif correlation is 'kendall':
        corr.append(kendalltau(c, train_target)[0])

sort_list = [list(i) for i in zip(*sorted(zip(index, np.abs(corr)),
                                          key=lambda x: x[1], reverse=cut))]

delf = sort_list[0][flen:]
cut_index, cut_corr = np.delete(index, delf), np.delete(corr, delf)
print(cut_index, cut_corr)
train_data = np.delete(train_data, delf, axis=1)

data = {}
traint = np.transpose(train_data)
for i, j in zip(traint, index):
    data[j] = i
df = pd.DataFrame(data)

ax = sns.violinplot(data=df, inner=None)
plt.show()
