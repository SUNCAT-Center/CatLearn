import numpy as np
from atoml.cross_validation import HierarchyValidation
from atoml.fit_funcs import find_optimal_regularization, RR

hv = HierarchyValidation(db_name='train_db.sqlite', table='FingerVector',
                         file_name='split')
hv.split_index(min_split=50, max_split=2000)
ind = hv.load_split()


def predict(train_features, train_targets, test_features, test_targets):
    data = {}
    b = find_optimal_regularization(X=train_features, Y=train_targets, p=0,
                                    Ns=100)
    coef = RR(X=train_features, Y=train_targets, p=0, omega2=b, W2=None,
              Vh=None)[0]

    # Test the model.
    sumd = 0.
    for tf, tt in zip(test_features, test_targets):
        sumd += (np.dot(coef, tf) - tt) ** 2
    error = (sumd / len(test_features)) ** 0.5

    data['result'] = len(test_targets), error

    return data


res = hv.split_predict(index_split=ind, predict=predict)
for i in res:
    print(i[0], i[1])
