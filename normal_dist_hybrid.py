from scipy.stats import multivariate_normal as MVN
from scipy.stats import t
from numpy.random import random
import numpy as np
from numpy.linalg import inv
from numpy import transpose as tp
from scipy.optimize import minimize
import warnings
import csv
import sys

"""
The prior density for the parameter. Here, mu ~ N_2(0, 500*I)
"""
def pi(mu):
    try:
        return MVN.pdf(mu, mean=[0, 0], cov = sigma_0)
    except RuntimeWarning as e:
        print("pi:", e)
        return 0


"""
The density of the data given the parameter.
x|mu ~ N_2(mu, Sigma) for known (constant) covariance matrix Sigma
"""
def f(x, mu):
    try:
        return MVN.pdf(x, mean=mu, cov = [[38, 0.8], [0.8, 4]])
    except RuntimeWarning as e:
        print("f:", e)
        return 0


np.random.seed(0)

sigma_0 = [[50, 0], [0, 50]]

# We generated 100 data points in R and saved them to the file normal_sample.csv
# This is, of course, a bad idea. Really, we should generate the data in Python itself.
# Note that the sample should be x_1, ..., x_100 \iid N([27\\13], [38 & 0.8 \\ 0.8 & 4])
n = 100

filename = "normal_sample"

sigma = np.array([[38, 0.8], [0.8, 4]])

data = list()
i = 0
with open("./datasets/" + filename + ".csv") as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        i += 1
        a, b = line[0].split(' ')
        data.append(np.array([float(a), float(b)]))
        if i == n:
            break

print("True params:")
print([inv(inv(sigma_0) + n*inv(sigma)) @ (n * inv(sigma) @ sum(data)/n),
    inv(inv(sigma_0) + n * inv(sigma))])

init_guess = [float(sys.argv[1]), float(sys.argv[2])]

chain = [init_guess]
for i in range(int(sys.argv[3])):
    X_prop = np.array([t.rvs(loc=chain[i][0], df=3), t.rvs(loc=chain[i][1], df=3)])

    acceptance_ratio = 1
    for x in data:
        acceptance_ratio *= f(x, X_prop)/f(x, chain[i])
    acceptance_ratio *= pi(X_prop)/pi(chain[i])
    acceptance_ratio = min([1, acceptance_ratio])

    if random() < acceptance_ratio:
        chain.append(X_prop)
    else:
        chain.append(chain[i])

chain = chain[-int(sys.argv[4]):]
if len(sys.argv) == 5:
    print("Estimated distribution")
    print(np.mean([c[0] for c in chain]), np.mean([c[1] for c in chain]))
    print(np.cov(chain, rowvar=False))
    exit()


# Initial guesses for variational parameters for CAVI
ls = [np.mean([c[0] for c in chain]), np.var([c[0] for c in chain]),
        np.mean([c[1] for c in chain]), np.var([c[1] for c in chain])]

np.seterr(all='warn')
warnings.filterwarnings('error')

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


def approximate_integral(func, bounds, precision = 3):
    points = np.mgrid[bounds[0][0]:bounds[0][1]:precision,
            bounds[1][0]:bounds[1][1]:precision].reshape(2,-1).T
    return sum([func(x, y)*precision**2 for (x, y) in points]), 0

def elbo(x, l1, l2, l3, l4, precision = 3):
    try:
        def inside(mu1, mu2):
            qm1m2 = q([mu1, mu2], l1, l2, l3, l4)
            return (sum([np.log(f(xi, [mu1, mu2])) for xi in x]) +
                     np.log(pi([mu1, mu2])) - np.log(qm1m2))*qm1m2
        SPREAD = 19 # Tune this as necessary
        result, abserr = approximate_integral(inside,
                [[l1-SPREAD*l2, l1+SPREAD*l2], [l3-SPREAD*l4, l3+SPREAD*l4]], precision = 1)
    except RuntimeWarning as e:
        print(e)
        exit() # Something went wrong: Abort.
    return result


# What range are we going to look for the variational parameter values
MEAN_RANGE = (10, 40)
VAR_RANGE = 5

# The CAVI algorithm
old_elbo = elbo(data, *ls, precision = 0.1)
while True:
    # Optimize q_1
    l1, l2 = minimize(lambda l: -1*elbo(data, l[0], l[1], ls[2], ls[3]), [ls[0], ls[1]],
            bounds=[MEAN_RANGE, (0.001, VAR_RANGE)], method="TNC", options={'xtol': 0.01}).x
    # Optimize q_2
    l3, l4 = minimize(lambda l: -1*elbo(data, l1, l2, l[0], l[1]), [ls[2], ls[3]],
            bounds=[MEAN_RANGE, (0.001, VAR_RANGE)], method="TNC", options={'xtol': 0.01}).x
    new_ls = [l1, l2, l3, l4]

    new_elbo =  elbo(data, *new_ls, precision = 0.1)
    if np.isclose(old_elbo, new_elbo):
        print("Final params:", new_ls)
        break
    ls = new_ls
    old_elbo = new_elbo
