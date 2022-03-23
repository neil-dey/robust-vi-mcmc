import numpy as np
from numpy.linalg import inv
from numpy import transpose as tp
import scipy.integrate as integrate
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import minimize
import warnings
import traceback
import csv
import sys

"""
The variational distribution. Here, q ~ N_2([l1, l2], [[l2, 0], [0, l4]]) which means that
q_1 ~ N(l1, l2)
q_2 ~ N(l3, l4)
"""
def q(z, l1, l2, l3, l4):
    try:
        return MVN.pdf(z, mean=[l1, l3], cov = [[l2, 0], [0, l4]])
    except RuntimeWarning as e:
        print(e)
        return 0


"""
The prior density for the parameter. Here, mu ~ N_2(0, 5000*I)
"""
def pi(mu):
    global sigma_0
    try:
        return MVN.pdf(mu, mean=[0, 0], cov = sigma_0)
    except RuntimeWarning as e:
        print(e)
        return 0

"""
The density of the data given the parameter.
x|mu ~ N_2(mu, Sigma) for known (constant) covariance matrix Sigma
"""
def f(x, mu):
    try:
        return MVN.pdf(x, mean=mu, cov = [[38, 0.8], [0.8, 4]])
    except RuntimeWarning as e:
        print(e)
        return 0

np.random.seed(0)
np.seterr(all='warn')
warnings.filterwarnings('error')


# We generated 100 data points in R and saved them to the file normal_sample.csv
# This is, of course, a bad idea. Really, we should generate the data in Python itself.
# Note that the sample should be x_1, ..., x_100 \iid N([27\\13], [38 & 0.8 \\ 0.8 & 4])
n = 100

filename = "normal_sample"

x = list()
i = 0
with open("./datasets/" + filename + ".csv") as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i += 1
        a, b = line[0].split(' ')
        x.append(np.array([float(a), float(b)]))
        if i == n:
            break


# Calculate the true posterior
sigma_0 = np.array([[50, 0], [0, 50]])
prior_det = np.linalg.det(sigma_0)
sigma = np.array([[38, 0.8], [0.8, 4]])
true_means = inv(inv(sigma_0) + n*inv(sigma)) @ (n * inv(sigma) @ sum(x)/n)
true_vars = inv(inv(sigma_0) + n * inv(sigma))
print("True Params:", [true_means[0], true_vars[0][0], true_means[1], true_vars[1][1]])
print("True covariance", true_vars[1][0])


# Initial guesses for variational parameters
ls = [float(param) for param in sys.argv[1:5]]

# The CAVI algorithm
inv_sigma = inv(sigma)
v11 = inv_sigma[0][0]
v12 = inv_sigma[0][1]
v22 = inv_sigma[1][1]

mu1_K = sum([v11*xi[0] + v12*xi[1] for xi in x])
mu2_K = sum([v22*xi[1] + v12*xi[0] for xi in x])

num_iters = 0
while True:
    num_iters += 1
    l1 = [ls[0], ls[1]]
    l2 = [ls[2], ls[3]]
    # Optimize q_1
    denom = n/2*v11 + 1/(2*prior_det**0.5)
    l1[0] = (mu1_K - n*v12*l2[0])/(2 * denom)
    l1[1] = 0.5/denom

    ls[0] = l1[0]
    ls[1] = l1[1]

    # Optimize q_2
    denom = n/2*v22 + 1/(2*prior_det**0.5)
    l2[0] = (mu2_K - n*v12*l1[0])/(2*denom)
    l2[1] = 0.5/denom

    new_ls = [l1[0], l1[1], l2[0], l2[1]]
    if np.allclose(new_ls, ls):
        print("Estimated params", new_ls)
        break
    ls = new_ls

