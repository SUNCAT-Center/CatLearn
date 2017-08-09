from cross_validation import HierarchyValidation
import seaborn as sns
import matplotlib.pyplot as plt


hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchey')
i = 21
lim = 25
subplot = 1
select_limit = [0, 40]
"""select_limit2=[i-1, i+1]"""


while i < lim:
    data_size, p_error, no_feature = hv.hierarcy(
       370, 199, 3200, hv, new_data=False,
       ridge=True, scale=True, globalscale=True, normalization=True,
       featselect_featvar=True, featselect_featconst=True,
       select_limit=select_limit, feat_sub=i)

    if not no_feature:
        plt.figure(1)
        plt.subplot(int("22"+str(subplot))).set_title('Feature size '+str(i),
                                                      loc='left')
        plt.legend(loc='upper right')
        plt.ylabel('Prediction error')
        plt.xlabel('Data size')
        sns.violinplot(x=data_size, y=p_error, scale="count")
        sns.pointplot(x=data_size, y=p_error, ci=100, capsize=.2)
        i += 1
        subplot += 1
    else:
        print("No subset "+str(i))
        i += 1
        lim += 1

plt.show()
