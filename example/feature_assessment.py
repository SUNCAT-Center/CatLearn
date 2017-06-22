"""Plot correlation between features."""
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import robust_scale

from atoml.cross_validation import HierarchyValidation
from atoml.feature_engineering import single_transform
from atoml.utilities import clean_variance

triangle = True  # Plot only the bottom half of the matrix
divergent = False  # Plot with a diverging color palette.
absolute = True  # Take the absolute values for the correlation.
clean = False  # Clean zero-varience features
expand = False

correlation = 'pearson'

# Define the hierarchey cv class method.
hv = HierarchyValidation(db_name='../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='split')
# Split the data into subsets.
hv.split_index(min_split=500, max_split=3000)
# Load data back in from save file.
ind = hv.load_split()
for i in ind:
    data = np.array(hv._compile_split(ind[i])[:, 1:-1], np.float64)
    if clean:
        data = clean_variance(data)['train']
    if expand:
        # Expand feature space to add single variable transforms.
        data = np.concatenate((data, single_transform(data)), axis=1)
    data = robust_scale(data)
    d = pd.DataFrame(data)

    # Compute the correlation matrix
    corr = d.corr(method=correlation)
    if absolute:
        corr = corr.abs()

    mask = None
    if triangle:
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    if not divergent:
        # Generate a custom diverging colormap
        cmap = sns.dark_palette((260, 75, 60), input="husl", as_cmap=True,
                                reverse=False)
    else:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap
    sns.heatmap(corr, cmap=cmap, mask=mask, square=True, xticklabels=20,
                yticklabels=20)
    plt.title(correlation)
    plt.xlabel('Feature No.')
    plt.ylabel('Feature No.')
    plt.show()
    exit()
