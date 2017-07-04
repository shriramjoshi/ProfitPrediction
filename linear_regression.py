import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# We can try alpha values in range [....., 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, ....]
alpha = 0.01

# Sometimes we need to take number of iterations if gradient descent is unable to find global optimum
iterations = 1500

dataset = pd.read_csv('example.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X = np.insert(X, 0, 1, axis=1)
y = np.transpose(np.matrix(y))
# Number of training examples
m = len(y)

theta = [[0], [0]]

def hypothesis(X, theta):
    # h0(X) = theta0 + theta1X (for one variable)
    h = np.dot(X, theta)
    return h

#COMPUTECOST Compute cost for linear regression
#   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#   parameter for linear regression to fit the data points in X and y
def costFunction(X, y, theta):
    # compute hypothesis
    h = hypothesis(X, theta)
    # compute sqaure error
    sqError = np.square(np.subtract(h, y))
    # compute cost-function for Linear Regression
    J = (1 / (2 * m)) * np.sum(sqError)
    return J

# GRADIENTDESCENT Performs gradient descent to learn theta
#   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
#   taking num_iters gradient steps with learning rate alpha
def gradientDescent(X, y, theta, alpha, iterations):
    JHistory = []
    newX = X[:, [1]]
    for i in range(0, iterations):
        h = theta[0] + ([theta[1]] * newX)
        # Calculate theta[0]
        theta_0 = theta[0] - alpha * (1 / m) * np.sum(np.subtract(h, y))
        # Calculate that[1]
        theta_1 = theta[1] - alpha * (1 / m) * np.sum(np.multiply((h - y), newX))
        theta[0] = theta_0
        theta[1] = theta_1
        JHistory.append(costFunction(X, y, theta))

    return theta


print('Theta values from Gradient Descent are : ')
theta = gradientDescent(X, y, theta, alpha, iterations);
print(theta)

print('Predict the profit for population size of 35000 & 70000')
prediction_1 = [1, 3.5] * theta
print('Prediction of the profit for population size of 35000 %', prediction_1);
prediction_2 = [1, 7] * theta
print('Prediction of the profit for population size of 70000 %', prediction_2);