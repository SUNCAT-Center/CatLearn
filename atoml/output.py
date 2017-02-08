import time

localtime = time.asctime(time.localtime(time.time()))


def write_output(testdata=None, traindata=None, prediction=None):
    with open('filelist.txt', 'w+') as myfile:
        myfile.write('Output from the AtoML code\nRun on %s\n' % localtime)
        if prediction is not None and 'validation_rmse' in prediction and \
           'average' in prediction['validation_rmse']:
            val = True
            myfile.write('\nAveraged validation error for the model is %s\n' %
                         prediction['validation_rmse']['average'])
        # Add the index for the training candidates.
        if traindata is not None and 'order' in traindata:
            index = traindata['order']
        else:
            index = None

        # Add the index for the test candidates.
        myfile.write('\nIndex stored for trainset:\n%s\n' % str(index))
        if testdata is not None and 'taken' in testdata:
            index = testdata['taken']
        else:
            index = None
        myfile.write('\nIndex stored for testset:\n%s\n' % str(index))

        if val:
            myfile.write('\nIndex stored for testset:\n%s\n' % str(index))
