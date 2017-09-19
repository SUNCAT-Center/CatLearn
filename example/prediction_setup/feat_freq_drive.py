"""Run the hierarchy to see which features occurs most frequent."""
from atoml.cross_validation import Hierarchy
from atoml.learning_curve import feature_frequency
from matplotlib import pyplot as plt
import numpy as np
from astropy.table import Table

hv = Hierarchy(db_name='../../data/train_db.sqlite', table='FingerVector',
               file_name='hierarchy')

# There exist two problem:
# 1. If no feature set is found it will get error. (Increase ref in FS)
# 2. Only a part of the list is shown (have fixed before, but not saved...)

vio = False
# Produce frequency plots between the lower and upp bound.
for i in range(20, 25):
    lim = i+4
    subplot = 1

    select_limit = [i-1, i+1]
    data1 = np.empty(1,)
    data2 = np.empty(1,)
    hit1, hit2 = 0, 0
    for k in range(1, 10):
        selected_features1 = feature_frequency(
            hv, 370, 3, 8, new_data=True, ridge=True, scale=True,
            globalscale=True, normalization=True, featselect_featvar=False,
            featselect_featconst=True, select_limit=select_limit, feat_sub=i)
        selected_features2 = feature_frequency(
            hv, 370, 3, 8, smallest=True, new_data=False, ridge=True,
            scale=True, globalscale=True, normalization=True,
            featselect_featvar=False, featselect_featconst=True,
            select_limit=select_limit, feat_sub=i)
        if bool(selected_features1):
            hit1 += 1
        if bool(selected_features2):
            hit2 += 1
        if bool(selected_features1) and bool(selected_features2):
            data1 = np.concatenate(
                (data1, (list(selected_features1.items())[0])[1][0][:]),
                axis=0)
            data2 = np.concatenate(
                (data2, (list(selected_features2.items())[0])[1][0][:]),
                axis=0)
    data1 = np.delete(data1, 0)
    data2 = np.delete(data2, 0)
    print(data1, data2)
    data_all = np.concatenate((data1, data2), axis=0)
    range1 = (int(min(data1)), int(max(data1)))
    range2 = (int(min(data2)), int(max(data2)))
    bins = np.arange(min(data_all)-2, max(data_all)+2, 0.5)
    hist1 = np.histogram(data1, bins=bins)
    hist2 = np.histogram(data2, bins=bins)
    r1_hist1 = np.delete(hist1[0], np.where(hist1[0] == 0))
    r1_hist1 = np.divide(r1_hist1.astype('float'), len(data1))*100
    r2_hist1 = np.delete(np.delete(hist1[1], np.where(hist1[0] == 0)), -1)

    r1_hist2 = np.delete(hist2[0], np.where(hist2[0] == 0))
    r1_hist2 = np.divide(r1_hist2.astype('float'), len(data2))*100
    r2_hist2 = np.delete(np.delete(hist2[1], np.where(hist2[0] == 0)), -1)

    if np.shape(r1_hist2)[0] > np.shape(r1_hist1)[0]:
        dif = np.shape(r1_hist2)[0] - np.shape(r1_hist1)[0]
        r1_hist1 = np.concatenate((r1_hist1, np.zeros(dif)), axis=0)
        r2_hist1 = np.concatenate((r2_hist1, np.zeros(dif)), axis=0)
    elif np.shape(r1_hist1)[0] > np.shape(r1_hist2)[0]:
        dif = np.shape(r1_hist1)[0] - np.shape(r1_hist2)[0]
        r1_hist2 = np.concatenate((r1_hist2, np.zeros(dif)), axis=0)
        r2_hist2 = np.concatenate((r2_hist2, np.zeros(dif)), axis=0)

    print("Feature set "+str(i))
    print("Dataset 2000            Dataset 250")
    print(Table([r2_hist1, np.around(r1_hist1, 3), r2_hist2,
          np.around(r1_hist2, 3)],
          names=('Feat_2000', 'Freq_2000 (%)',
          'Feat_250', 'Freq_250 (%)')))

    fig = plt.figure(figsize=(10, 10))

    plt.hist(data1, bins=bins, range=range1,  ls='dashed',
             alpha=0.5, lw=3, color='b',
             label='Small dataset 249 ('+str(hit1)+')')
    plt.hist(data2, bins=bins, range=range2, ls='dotted', alpha=0.5,
             lw=3, color='r', label='Large dataset 2000('+str(hit2)+')')
    plt.legend(loc='upper right')
    plt.title('Feature set '+str(i))
    plt.ylabel('Frequency')
    plt.xlabel('Feature')
    plt.show()
