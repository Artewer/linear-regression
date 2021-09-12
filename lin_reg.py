import os
import inline as inline
import matplotlib
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
import utils


def warmUpExercise():
    """
    Example function in Python which computes the identity matrix.

    Returns
    -------
    A : array_like
        The 5x5 identity matrix.

    Instructions
    ------------
    Return the 5x5 identity matrix.
    """
    # ======== YOUR CODE HERE ======
    A = np.eye(5)

    # ==============================
    return A


def plotData(x, y):
    """
    Plots the data points x and y into a new figure. Plots the data
    points and gives the figure axes labels of population and profit.

    Parameters
    ----------
    x : array_like
        Data point values for x-axis.

    y : array_like
        Data point values for y-axis. Note x and y should have the same size.

    Instructions
    ------------
    Plot the training data into a figure using the "figure" and "plot"
    functions. Set the axes labels using the "xlabel" and "ylabel" functions.
    Assume the population and revenue data have been passed in as the x
    and y arguments of this function.

    Hint
    ----
    You can use the 'ro' option with plot to have the markers
    appear as red circles. Furthermore, you can make the markers larger by
    using plot(..., 'ro', ms=10), where `ms` refers to marker size. You
    can also set the marker edge color using the `mec` property.
    """
    fig = pyplot.figure()  # open a new figure

    # ====================== YOUR CODE HERE =======================
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Profit in 10,000$')
    pyplot.xlabel('Population of the city in 10,000s')
    pyplot.plot(X[:, 1], np.dot(X, theta), '-')
    pyplot.legend(['Training data', 'Linear regression'])

    pyplot.show()

    # =============================================================


def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression. Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1), where m is the number of examples,
        and n is the number of features. We assume a vector of one's already
        appended to the features so we have n+1 columns.

    y : array_like
        The values of the function at each data point. This is a vector of
        shape (m, ).

    theta : array_like
        The parameters for the regression function. This is a vector of
        shape (n+1, ).

    Returns
    -------
    J : float
        The value of the regression cost function.

    Instructions
    ------------
    Compute the cost of a particular choice of theta.
    You should set J to the cost.
    """

    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE =====================
    f_size = theta.size
    for i in range(m):
        h = 0
        for i2 in range(f_size):
            h += X[i][i2] * theta[i2]
        J += (h - y[i]) ** 2
    J = J / (m * 2)

    # ===========================================================
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.

    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).

    y : array_like
        Value at given features. A vector of shape (m, ).

    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).

    alpha : float
        The learning rate.

    num_iters : int
        The number of iterations for gradient descent.

    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).

    J_history : list
        A python list for the values of the cost function after each iteration.

    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.

    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()

    J_history = []  # Use a python list to save cost in every iteration
    f_size = theta.size
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        o1 = 0
        o2 = 0
        f_size = theta.size
        for i in range(m):
            h = 0
            for i2 in range(f_size):
                h += X[i][i2] * theta[i2]
            o1 += (h - y[i])
            o2 += (h - y[i]) * X[i][1]
        theta[0] = theta[0] - alpha * o1 / m
        theta[1] = theta[1] - alpha * o2 / m

        # =====================================================================

        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).

    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).

    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.

    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.

    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    m = len(X_norm)
    # =========================== YOUR CODE HERE =====================
    mu = np.mean(X_norm, axis=0)
    sigma = np.std(X_norm, axis=0)
    for i in range(m):
        for feature in range(X_norm[0].size):
            X_norm[i][feature] -= mu[feature]
            X_norm[i][feature] /= sigma[feature]
    # ================================================================
    return X_norm, mu, sigma


def gradientDescentMulti(X, y, theta, alpha, num_iters):

    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()

    J_history = []
    f_size = theta.size

    for i in range(num_iters):
        theta_val = np.zeros(f_size)
        for i in range(m):
            h = 0
            for i2 in range(f_size):
                h += X[i][i2] * theta[i2]
            for i3 in range(f_size):
                change = (h - y[i]) * X[i][i3]
                theta_val[i3] += change
        for i in range(f_size):
            theta[i] = theta[i] - alpha * theta_val[i] / m
        #print(theta)
        print(computeCostMulti(X,y,theta))
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history


def normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).

    y : array_like
        The value at each data point. A vector of shape (m, ).

    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).

    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.

    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])

    # ===================== YOUR CODE HERE ============================
    Xt = np.transpose(X)
    inverted = np.linalg.pinv(np.matmul(Xt, X))
    last = np.matmul(inverted, Xt)
    theta = np.matmul(last, y)

    # =================================================================
    return theta


data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size  # number of training examples


X = np.stack([np.ones(m), X], axis=1)

theta = np.zeros(2)
iterations = 1500
alpha = 0.01

theta, J_history = gradientDescentMulti(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')

plotData(X[:, 1], y)
# Predict values for population sizes of 35,000 and 70,000

predict1 = np.dot([1, 3.5], theta)
print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2*10000))

