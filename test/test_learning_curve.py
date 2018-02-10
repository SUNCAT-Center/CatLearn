"""Run the hierarchy with feature selection."""
import numpy as np
from astropy.table import Table

from atoml.cross_validation import Hierarchy
from atoml.learning_curve import hierarchy, feature_frequency


def learning_curve_test():
    hv = Hierarchy(db_name='data/train_db.sqlite', file_name='hierarchy')

    # If you want to keep datasize fixed, and vary features.
    featselect_featvar = False
    # If False above. Feature selection for ordinary ridge regression, or just
    # plain vanilla ridge.
    featselect_featconst = True

    # i is the size of featureset.
    i = 10
    lim = i+2
    if featselect_featvar:
        # Under which interval of feature set in inspection.
        # (Varying of feature size)
        select_limit = [0, 20]
    else:
        # For which fetureset inspection is made. (Varying of data size)
        select_limit = [i-1, i+1]
    while i < lim:
        set_size, p_error, result, PC = hierarchy(
            hv, 370, 10, 50, new_data=True, ridge=True, scale=True,
            globalscale=True, normalization=True,
            featselect_featvar=featselect_featvar,
            featselect_featconst=featselect_featconst,
            select_limit=select_limit,
            feat_sub=i)
        if not (set_size and p_error) == [] and not featselect_featvar:
            for data in result:
                print('data size:', data[0], 'prediction error:',
                      data[1], 'Omega:', data[5],
                      'Euclidean length:', data[2],
                      'Pearson correlation:', data[3])
            i += 1
        elif (set_size and p_error) == [] and not featselect_featvar:
            print("No subset {}".format(i))
            i += 1
            lim += 1
        if featselect_featvar:
            # Don't want to make four subpl for varying of features.
            i += 4
        select_limit = [i-1, i+1]


def frequency_test():
    hv = Hierarchy(db_name='data/train_db.sqlite', file_name='hierarchy')
    # Produce frequency plots between the lower and upp bound.
    for i in range(20, 22):

        select_limit = [i-1, i+1]
        data1 = np.empty(1,)
        data2 = np.empty(1,)
        hit1, hit2 = 0, 0
        for k in range(1, 2):
            selected_features1 = feature_frequency(
                hv, 370, 3, 8, new_data=True, ridge=True, scale=True,
                globalscale=True, normalization=True, featselect_featvar=False,
                featselect_featconst=True, select_limit=select_limit,
                feat_sub=i)
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

        data_all = np.concatenate((data1, data2), axis=0)
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


if __name__ == '__main__':
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    learning_curve_test()
    frequency_test()

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))
