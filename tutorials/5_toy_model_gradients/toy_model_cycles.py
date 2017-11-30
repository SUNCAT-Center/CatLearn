""" This tutorial is intended to help you get familiar with using AtoML to set
up a model and do predictions.

First we set up a known underlying function in one dimension (including
first derivative). Then, we pick some values to train.
Finally we will use AtoML to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
"""
import numpy as np
import matplotlib.pyplot as plt
from atoml.preprocess.feature_preprocess import standardize
from atoml.preprocess.scale_target import target_standardize
from atoml.regression import GaussianProcess
from atoml.utilities.cost_function import get_error


# The user can choose whether the features and/or the targets are standardized.
StandardizeFeatures = True
StandardizeTargets = True

# First derivative observations can be included.
eval_gradients = True
number_of_iterations = 11

# A known underlying function in one dimension [y] and first derivative [dy].
def afunc(x):
    """ Function [y] and first derivative [dy] """
    y =  3.0+np.sin(x)*np.cos(x)*np.exp(2*x)*x**-2*np.exp(-x)*np.cos(
    x)*np.sin(x)
    dy =  (2 * np.exp(x)*np.cos(x)**3* np.sin(x))/x**2-(2*np.exp(x)* np.cos(
    x)**2* np.sin(x)**2)/x**3+(np.exp(x)*np.cos(x)**2 *np.sin(x)**2)/x**2-(
    2*np.exp(x)*np.cos(x)*np.sin(x)**3)/x**2
    return [y,dy]

# Define initial and final state.
train = np.array([[0.01],[5.9]])

# Define initial prediction parameters.
reg = 0.01
w1 = 1.0  # Too large widths results in a biased model.
scaling = 1.0
scaling_const = 1.0
constant = 1.0
fig = plt.figure(figsize=(13.0, 6.0))

for iteration in range(1,number_of_iterations):
    number_of_plot = iteration

    # Setting up data.

    # A number of training points in x.
    # Each element in the list train can be referred to as a fingerprint.

    # Call the underlying function to produce the target values.
    target = np.array(afunc(train)[0])

    # Generate test datapoints x.
    test_points = 150
    test = np.linspace(0.1,6.0,test_points)
    test = np.reshape(test, (test_points, 1))

    # Make a copy of the original features and targets.

    org_train = train.copy()
    org_target = target.copy()
    org_test = test.copy()

    # Standardization of the train, test and target data.

    if StandardizeFeatures:
        feature_std = standardize(train_matrix=train,test_matrix=test)
        train, train_mean, train_std = feature_std['train'], feature_std['mean'], \
        feature_std['std']
        test = (test -train_mean)/train_std

    else:
        train_mean, train_std = 0.0, 1.0
    if StandardizeTargets:
        target_std = target_standardize(target)
        target, target_mean, target_std = target_std['target'], target_std[
        'mean'], target_std['std']
    else:
        target_mean, target_std = 0.0, 1.0

    # Call the underlying function to produce the gradients of the target values.

    if eval_gradients:
        gradients = []
        for i in org_train:
            gradients.append(afunc(i)[1])
        org_gradients = np.asarray(gradients)
        gradients = org_gradients/(target_std/train_std)
        y_tilde = np.append(target, gradients)
        target = np.reshape(y_tilde,(np.shape(y_tilde)[0],1))

    # Gaussian Process.


    # Set up the prediction routine and optimize hyperparameters.
    kdict = {'k1': {'type': 'gaussian', 'width': w1, 'scaling': scaling}}

    kdict = {'k1': {'type': 'gaussian', 'width': w1, 'scaling': scaling},
    'k2': {
    'type': 'linear', 'scaling': scaling}}

    # kdict = {'k1': {'type': 'gaussian', 'width': w1, 'scaling': scaling},
    # 'k2': {
    # 'type': 'constant','const': constant, 'scaling': scaling_const}}

    gp = GaussianProcess(kernel_dict=kdict, regularization=reg**2,
                         train_fp=train,
                         train_target=target,
                         optimize_hyperparameters=True,
                         eval_gradients=eval_gradients, algomin='L-BFGS-B',
                         global_opt=False)
    print('Optimized kernel:', gp.kernel_dict)


    # Do the optimized predictions.
    pred = gp.predict(test_fp=test, uncertainty=True)
    prediction = np.array(pred['prediction'][:,0])

    # Calculate the uncertainty of the predictions.
    uncertainty = np.array(pred['uncertainty'])

    # Get confidence interval on predictions.
    upper = prediction + uncertainty
    lower = prediction - uncertainty

    # Scale predictions back to the original scale.
    if StandardizeTargets:
        uncertainty = (uncertainty*target_std) + target_mean
        prediction = (prediction*target_std) + target_mean
        upper = (upper*target_std) + target_mean
        lower = (lower*target_std) + target_mean

    # Get average errors.
    error = get_error(prediction, afunc(test)[0])
    print('Gaussian linear regression prediction:', error['absolute_average'])

    # Add new point:
    # 1) Calculate the gradients of the predicted function.
    # 2) Get the points were the gradients are below grad_prediction_interval
    # 3) For these points get the point which has the maximum uncertainty.
    # 4) Add it to the next train list.

    grad_prediction = np.abs(np.gradient(prediction))
    grad_prediction_interval = (np.max(grad_prediction)-np.min(
    grad_prediction))/1e10
    while not np.all(grad_prediction<grad_prediction_interval):
        grad_prediction_interval = grad_prediction_interval*10.0
    index = (np.linspace(1,len(prediction)/1,len(prediction/1))-1)/1
    index_grad = index[grad_prediction<grad_prediction_interval]
    index_new = np.int(index_grad[np.argmax(uncertainty[
    grad_prediction<grad_prediction_interval])])

    new_train_point = org_test[index_new]
    new_train_point = np.reshape(new_train_point,(np.shape(new_train_point)[0],1))
    train = np.concatenate((org_train,new_train_point))

    # Update hyperarameters with the optimised ones after n iterations.

    if iteration > 3:
        reg = gp.regularization
        w1 = gp.kernel_dict['k1']['width']
        scaling = gp.kernel_dict['k1']['scaling']
        constant = gp.kernel_dict['k2']['const']
        scaling_const = gp.kernel_dict['k2']['scaling']

    # Plotting.

    # Store the known underlying function for plotting.

    linex = np.linspace(0.1,6.0,test_points)
    linex = np.reshape(linex, (1,np.shape(linex)[0]))
    linex = np.sort(linex)
    liney = []
    for i in linex:
        liney.append(afunc(i)[0])


    # Example
    ax = fig.add_subplot(2,5,number_of_plot)
    ax.plot(linex[0], liney[0], '-', lw=1, color='black')
    ax.plot(org_train, org_target, 'o', alpha=0.2, color='black')
    ax.plot(org_test, prediction, 'g-', lw=1, alpha=0.4)
    ax.fill_between(org_test[:,0], upper, lower, interpolate=True,
    color='blue', alpha=0.2)
    plt.title('GP iteration'+str(number_of_plot),fontsize=9)
    # plt.xlabel('Descriptor')
    # plt.ylabel('Response')
    # plt.title('Iteration')
    plt.axis('tight')
    plt.xticks(fontsize = 6)

    # Gradients

    if eval_gradients==True:
        size_bar_gradients = (np.abs(np.max(linex) - np.min(linex))/2.0)/25.0
        def lineary(m,linearx,train,target):
                """Define some linear function."""
                lineary = m*(linearx-train)+target
                return lineary
        for i in range(0,np.shape(org_gradients)[0]):
            linearx_i = np.linspace(org_train[i]-size_bar_gradients, org_train[i]+
            size_bar_gradients,num=10)
            lineary_i = lineary(org_gradients[i],linearx_i,org_train[i],
            org_target[i])
            ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=0.5, color='black')


plt.show()







