"""This tutorial is intended to help you get familiar with using AtoML to set
up a model and do predictions.

First we set up a known underlying function in one dimension.
Then, we pick some values to train.
Finally we will use AtoML to make predictions on some unseen fingerprint and
benchmark those predictions against the known underlying function.
In this toy model we show that one can improve the resulting estimates by
including first derivative observations.
"""

import numpy as np
import matplotlib.pyplot as plt
from atoml.regression import GaussianProcess
from atoml.regression.cost_function import get_error


# A known underlying function in one dimension [y] and first derivative [dy].
def afunc(x):
    """Function [y] and first derivative [dy]."""
    y = 3.0+np.sin(x)*np.cos(x)*np.exp(2*x)*x**-2*np.exp(-x)*np.cos(x) * \
        np.sin(x)
    dy = (2*np.exp(x)*np.cos(x)**3*np.sin(x))/x**2 - \
        (2*np.exp(x)*np.cos(x)**2*np.sin(x)**2)/x**3 + \
        (np.exp(x)*np.cos(x)**2*np.sin(x)**2)/x**2 - \
        (2*np.exp(x)*np.cos(x)*np.sin(x)**3)/x**2
    return [y, dy]


# Define initial and final state.
train = np.array([[0.01], [6.0]])

# Define initial prediction parameters.
reg = 0.01
w1 = 1.0  # Too large widths results in a biased model.
scaling_exp = 1.0
constant = 1.0

# Create figure.
fig = plt.figure(figsize=(13.0, 6.0))

# Times that we train the model, in each iteration we add a new training point.
number_of_iterations = 10

for iteration in range(1, number_of_iterations+1):
    number_of_plot = iteration

    # Setting up data.

    # Call the underlying function to produce the target values.
    target = np.array(afunc(train)[0])

    # Generate test datapoints x.
    test_points = 500
    test = np.linspace(0.1, 6.0, test_points)
    test = np.reshape(test, (test_points, 1))

    # Make a copy of the original features and targets.
    org_train = train.copy()
    org_target = target.copy()
    org_test = test.copy()

    # Call underlying function to produce the gradients of the target values.
    gradients = []
    for i in org_train:
        gradients.append(afunc(i)[1])
    org_gradients = np.asarray(gradients)

    # Gaussian Process.

    # Set up the prediction routine and optimize hyperparameters.
    """" Note that in this example we used a combination of two kernels (squared
    # exponential and constant kernels)."""

    kdict = {'k1': {'type': 'gaussian', 'width': w1, 'scaling': scaling_exp},
             'k2': {'type': 'constant', 'const': constant,
                    }
             }

    gp = GaussianProcess(
        kernel_dict=kdict, regularization=reg**2, train_fp=train,
        train_target=target, optimize_hyperparameters=True,
        gradients=gradients,scale_data=True)
    print('Optimized kernel:', gp.kernel_dict)

    # Do the optimized predictions.
    pred = gp.predict(test_fp=test, uncertainty=True)
    prediction = np.array(pred['prediction'][:, 0])

    # Calculate the uncertainty of the predictions.
    uncertainty = np.array(pred['uncertainty'])

    # Get confidence interval on predictions.
    upper = prediction + uncertainty
    lower = prediction - uncertainty

    # Get average errors.
    error = get_error(prediction, afunc(test)[0])
    print('Gaussian linear regression prediction:', error['absolute_average'])

    """A new training point is added following:

    Firstly, we calculate the gradients of the predicted function.

    Secondly, we ask for the points which the value of their gradients are
    below grad_prediction_interval.

    Thirdly, within the previous set of points we ask for the one which
    has the maximum uncertainty.

    Finally, the "interesting" point is added to the list of training
    points."""

    grad_prediction = np.abs(np.gradient(prediction))
    grad_prediction_interval = (np.max(grad_prediction) -
                                np.min(grad_prediction))/1e10
    while not np.all(grad_prediction < grad_prediction_interval):
        grad_prediction_interval = grad_prediction_interval*10.0
    index = (np.linspace(1, len(prediction)/1, len(prediction/1))-1)/1
    index_grad = index[grad_prediction < grad_prediction_interval]
    index_new = np.int(index_grad[np.argmax(
        uncertainty[grad_prediction < grad_prediction_interval])])

    new_train_point = org_test[index_new]
    new_train_point = np.reshape(
        new_train_point, (np.shape(new_train_point)[0], 1))
    train = np.concatenate((org_train, new_train_point))

    # Update hyperarameters with the optimised ones after "n" iterations.

    if iteration > 5:
        reg = gp.regularization
        w1 = gp.kernel_dict['k1']['width']
        scaling_exp = float(gp.kernel_dict['k1']['scaling'])
        constant = float(gp.kernel_dict['k2']['const'])


    # Plots.

    # Store the known underlying function for plotting.

    linex = np.linspace(0.1, 6.0, test_points)
    linex = np.reshape(linex, (1, np.shape(linex)[0]))
    linex = np.sort(linex)
    liney = []
    for i in linex:
        liney.append(afunc(i)[0])

    # Example
    ax = fig.add_subplot(2, 5, number_of_plot)
    ax.plot(linex[0], liney[0], '-', lw=1, color='black')
    ax.plot(org_train, org_target, 'o', alpha=0.2, color='black')
    ax.plot(org_test, prediction, 'g-', lw=1, alpha=0.4)
    ax.fill_between(org_test[:, 0], upper, lower, interpolate=True,
                    color='blue', alpha=0.2)
    plt.title('GP iteration'+str(number_of_plot), fontsize=9)
    # plt.xlabel('Descriptor')
    # plt.ylabel('Response')
    # plt.title('Iteration')
    plt.axis('tight')
    plt.xticks(fontsize=6)

    # Gradients

    if gradients is not None:
        size_bar_gradients = (np.abs(np.max(linex) - np.min(linex))/2.0)/25.0

        def lineary(m, linearx, train, target):
                """Define some linear function."""
                lineary = m*(linearx-train)+target
                return lineary

        for i in range(0, np.shape(org_gradients)[0]):
            linearx_i = np.linspace(
                org_train[i]-size_bar_gradients,
                org_train[i]+size_bar_gradients, num=10)
            lineary_i = lineary(org_gradients[i], linearx_i, org_train[i],
                                org_target[i])
            ax.plot(linearx_i, lineary_i, '-', lw=3, alpha=0.5, color='black')

plt.show()
