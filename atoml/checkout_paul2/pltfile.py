import seaborn as sns
import matplotlib.pyplot as plt

def violinplot(set_size, p_error, subplot, i):
    """Data_process setup.

    Parameters
    ----------
    set_size : list
       Size of sub-set of data/features which the model is based on.
    p_error : list
       The prediction error for plain vanilla ridge.
    subplot : int
        Which subplot being produced.
    i : int
       Which iteration in the featureselection.
    """

    plt.figure(1)
    plt.subplot(int("22"+str(subplot))).set_title('Feature size '+str(i),
                                                  loc='left')
    plt.legend(loc='upper right')
    plt.ylabel('Prediction error')
    plt.xlabel('Data size')
    sns.violinplot(x=set_size, y=p_error, scale="count")
    sns.pointplot(x=set_size, y=p_error, ci=100, capsize=.2)
    if subplot == 4:
        plt.show()


def originalplot(set_size, p_error, PC, subplot, i):
    """Data_process setup.

    Parameters
    ----------
    set_size : list
       Size of sub-set of data/features which the model is based on.
    p_error : list
       The prediction error for plain vanilla ridge.
    PC : object
       Used to get statistics.
    subplot : int
        Which subplot being produced.
    i : int
       Which iteration in the featureselection.
    """
    p_error_mean_list, set_size_mean_list, corrected_std =\
        PC.get_statistic(set_size, p_error)
    plt.figure(1)
    plt.subplot(int("22"+str(subplot))).set_title('Feature size '+str(i),
                                                  loc='left')
    plt.legend(loc='upper right')
    plt.ylabel('Prediction error')
    plt.xlabel('Data size')
    plt.scatter(set_size, p_error, s=10, c='b', marker="s",
                label='Predicted error')
    plt.scatter(set_size_mean_list, p_error_mean_list, s=10, c='r', marker="o",
                label='Mean predicted error')
    plt.plot(set_size_mean_list, p_error_mean_list, '-o', color='red')
    plt.errorbar(set_size_mean_list, p_error_mean_list, corrected_std,
                 color='red')
    if subplot == 4:
        plt.show()
