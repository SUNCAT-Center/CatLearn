"""Plot correlation between features and target values."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

from atoml.cross_validation import HierarchyValidation
from atoml.feature_preprocess import standardize

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

# Scale and shape the data.
train_data = standardize(train_matrix=train_data)['train']
train_target = train_target.reshape(len(train_target), )

# Find the correlation.
corr = []
for c in train_data.T:
    if correlation is 'pearson':
        corr.append(pearsonr(c, train_target)[0])
    elif correlation is 'spearman':
        corr.append(spearmanr(c, train_target)[0])
    elif correlation is 'kendall':
        corr.append(kendalltau(c, train_target)[0])
ind = list(range(len(corr)))

# Plot the figure.
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.plot(ind, np.abs(corr), '-', color='black', lw=1)
ax.axis([0, len(ind), 0, 1])
plt.title(correlation)
plt.xlabel('Feature No.')
plt.ylabel('Correlation')

plt.show()
