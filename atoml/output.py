import time

localtime = time.asctime(time.localtime(time.time()))


def write_output(testatoms=None, trainatoms=None, prediction=None, testfp=None,
                 trainfp=None):
    with open('ATOMLout.txt', 'w+') as myfile:
        myfile.write('Output from the AtoML code\nRun on %s\n' % localtime)
        if prediction is not None and 'validation_rmse' in prediction and \
           'average' in prediction['validation_rmse']:
            val = True
            myfile.write('\nAveraged validation error for the model is %s\n' %
                         prediction['validation_rmse']['average'])
        # Add the index for the training candidates.
        if trainatoms is not None and 'order' in trainatoms:
            index = trainatoms['order']
        else:
            index = None

        # Add the index for the test candidates.
        myfile.write('\nIndex stored for trainset:\n%s\n' % str(index))
        if testatoms is not None and 'taken' in testatoms:
            index = testatoms['taken']
        else:
            index = None
        myfile.write('\nIndex stored for testset:\n%s\n' % str(index))

        if val:
            myfile.write('\nIndex stored for testset:\n%s\n' % str(index))
