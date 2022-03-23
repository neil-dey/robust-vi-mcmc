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
Here, q ~ N_2([l1, l2], [[l2, 0], [0, l4]]) which means that
q_1 ~ N(l1, l2)
q_2 ~ N(l3, l4)
"""
def q(z, l1, l2, l3, l4):
    try:
        return MVN.pdf(z, mean=[l1, l3], cov = [[l2, 0], [0, l4]])
    except RuntimeWarning as e:
        return 0


"""
Just a Riemann sum approximation for an integral.
You can be more accurate by choosing a better approximation scheme
"""
def approximate_integral(func, bounds, precision = 3):
    points = np.mgrid[bounds[0][0]:bounds[0][1]:precision,
            bounds[1][0]:bounds[1][1]:precision].reshape(2,-1).T
    return sum([func(x, y)*precision**2 for (x, y) in points]), 0

"""
The prior density for the parameter. Here, mu ~ N_2(0, 5000*I)
"""
def pi(mu):
    try:
        return MVN.pdf(mu, mean=[0, 0], cov = [[5000, 0], [0, 5000]])
    except RuntimeWarning as e:
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

"""
ELBO. Rewritten to prevent underflows as you calculate logs of
products of probability distributions
(mu1, mu2) is integrated over the range [0, 30] x [0, 30] to avoid pdfs being zero
"""
def elbo(x, l1, l2, l3, l4, precision = 3):
    try:
        def inside(mu1, mu2):
            qm1m2 = q([mu1, mu2], l1, l2, l3, l4)
            return (sum([np.log(f(xi, [mu1, mu2])) for xi in x]) +
                     np.log(pi([mu1, mu2])) - np.log(qm1m2))*qm1m2
        result, abserr = approximate_integral(inside, [[0, 30], [0, 30]], precision = 3)
        # Technically better but seems to work slower and doesn't seem to improve much at all
        #result, abserr = approximate_integral(inside,
            #[[l1-20*l2, l1+20*l2], [l3-20*l4, l3+20*l4]], precision = 3)
    except RuntimeWarning as e:
        return -1000 #In case a log(0) does occur, just return a really low ELBO
    return result


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
sigma = np.array([[38, 0.8], [0.8, 4]])
print("True params:")
print([inv(inv(sigma_0) + n*inv(sigma)) @
    (n * inv(sigma) @ sum(x)/n), inv(inv(sigma_0) + n * inv(sigma))])


# Initial guesses for variational parameters
ls = [float(param) for param in sys.argv[1:5]]
# What range are we going to look for the variational parameter values
MEAN_RANGE = (10, 30)
VAR_RANGE = 5

# The CAVI algorithm
old_elbo = elbo(x, *ls, precision = 0.1)
while True:
    # Optimize q_1
    l1, l2 = minimize(lambda l: -1*elbo(x, l[0], l[1], ls[2], ls[3]), [ls[0], ls[1]],
            bounds=[MEAN_RANGE, (0.01, VAR_RANGE)], method="TNC", options={'xtol': 0.1}).x
    # Optimize q_2
    l3, l4 = minimize(lambda l: -1*elbo(x, l1, l2, l[0], l[1]), [ls[2], ls[3]],
            bounds=[MEAN_RANGE, (0.01, VAR_RANGE)], method="TNC", options={'xtol': 0.1}).x
    new_ls = [l1, l2, l3, l4]

    new_elbo =  elbo(x, *new_ls, precision = 0.1)
    if np.isclose(old_elbo, new_elbo):
        print("Final params:", new_ls)
        break
    ls = new_ls
    old_elbo = new_elbo
