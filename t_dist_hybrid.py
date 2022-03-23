from scipy.stats import multivariate_normal as MVN
from scipy.stats import t
from scipy.stats import norm as UVN
from scipy.stats import gamma
from numpy.random import random
import numpy as np
from numpy.linalg import inv
from numpy import transpose as tp
from scipy.optimize import minimize
import warnings
import csv
import sys
from time import time

# Seed 2 has a full MCMC run that converges nicely within the number of steps we gave.
# You can use other seeds, but treating their results as a "ground truth" may be dubious
# Also, the MCMC results for this seed were saved ahead of time to compute KL divergences
SEED = 2
np.random.seed(SEED)

"""
Generates data from a multivariate t distribution using a Gaussian copula (covariance sigma).
Assumes sigma_ii = 1 for each i
"""
def generate_multi_t(nu, sigma = None):
    n = len(nu)
    if sigma is None:
        sigma = np.eye(n)

    data = UVN.cdf(MVN.rvs(cov=sigma))
    for i in range(n):
        data[i] = t.ppf(data[i], df = nu[i])

    return data

"""
The prior density for the parameter. Here, 1/nu ~ Unif(0, 1/2)
"""
def pi(nu):
    product = 1
    for nu_i in nu:
        if nu_i >= 2:
            product *= 2/(nu_i**2)
        else:
            return 0
    return product


def copula_density(u1, u2, det, RinvMinusI):
    return 1/det**0.5 * np.exp(-0.5 * np.array([UVN.ppf(u1), UVN.ppf(u2)]).T @ RinvMinusI @ np.array([UVN.ppf(u1), UVN.ppf(u2)]))

"""
The density of a data point given the parameter.
x|nu ~ t_2(nu, sigma) for known (constant) gaussian copula covariance matrix Sigma
"""
def f(x, nu, det, sigmaInvMinusI):
    return copula_density(t.cdf(x[0], df=nu[0]), t.cdf(x[1], df=nu[1]), det, sigmaInvMinusI) * t.pdf(x[0], df=nu[0]) * t.pdf(x[1], df=nu[1])


# Generate some data
n = 100
true_nu = [8, 50]
true_sigma = [[1, 0.5], [0.5, 1]]
detSigma = np.linalg.det(true_sigma)
RinvMinusI = inv(true_sigma) - np.eye(len(true_sigma))
data = [generate_multi_t(true_nu, sigma=true_sigma) for _ in range(n)]


# Figure what we are running
skip_MCMC = False
stop_post_MCMC = False
if sys.argv[1] == "MCMC":
    init_guess = [4, 4]
    num_steps = 100000
    post_burnin = 90000
    stop_post_MCMC = True
elif sys.argv[1] == "Hybrid":
    init_guess = [float(sys.argv[2]), float(sys.argv[3])]
    print(init_guess)
    num_steps = 500
    post_burnin = 100
else:
    ls = [float(param) for param in sys.argv[2:6]]
    skip_MCMC = True

start_time = time()
if not skip_MCMC:
    print("Performing MCMC")
    chain = [init_guess]
    for i in range(num_steps):
        X_prop = [abs(t.rvs(loc=chain[i][j], df=3)-2)+2 for j in range(len(true_nu))]
        acceptance_ratio = 1
        for x in data:
            acceptance_ratio *= f(x, X_prop, detSigma, RinvMinusI)/f(x, chain[i], detSigma, RinvMinusI)
        acceptance_ratio *= pi(X_prop)/pi(chain[i])
        acceptance_ratio = min([1, acceptance_ratio])

        if random() < acceptance_ratio:
            chain.append(X_prop)
        else:
            chain.append(chain[i])

    chain = chain[-post_burnin:]


    # Initial guesses for variational parameters for CAVI
    mean0 = np.mean([c[0] for c in chain])
    var0 = np.var([c[0] for c in chain], ddof=1)
    mean1 = np.mean([c[1] for c in chain])
    var1 = np.var([c[1] for c in chain], ddof=1)

    ls = [(mean0-2)**2/var0, var0/(mean0 - 2), (mean1-2)**2/var1, var1/(mean1-2)]

    print("MCMC took time", time() - start_time)
    if stop_post_MCMC:
        with open("./datasets/full_MCMC_results.csv", "w") as f:
            csv.writer(f).writerows(chain)
        exit()

np.seterr(all='warn')
warnings.filterwarnings('error')

"""
Here, we have independent
q_1 ~ Gamma(l1, l2) + 2
q_2 ~ Gamma(l3, l4) + 2
"""
def q(z, l1, l2, l3, l4):
    try:
        return gamma.pdf(z[0], l1, loc=2, scale=l2) * gamma.pdf(z[1], l3, loc=2, scale=l4)
    except RuntimeWarning as e:
        return 0


def approximate_integral(func, bounds, precision = 3):
    points = np.mgrid[bounds[0][0]:bounds[0][1]:precision,
            bounds[1][0]:bounds[1][1]:precision].reshape(2,-1).T
    return sum([func(x, y)*precision**2 for (x, y) in points]), 0

def elbo(x, l1, l2, l3, l4, precision = 3):
    try:
        def inside(nu1, nu2):
            qm1m2 = q([nu1, nu2], l1, l2, l3, l4)
            if(qm1m2 == 0):
                return 0
            return (sum([np.log(f(xi, [nu1, nu2], detSigma, RinvMinusI)) for xi in x]) +
                     np.log(pi([nu1, nu2])) - np.log(qm1m2))*qm1m2
        result, abserr = approximate_integral(inside,
                [[2.01, 100], [2.01, 100]], precision)
    except RuntimeWarning as e:
        print(e)
        exit() # Something went wrong: Abort.
    return result


# What range are we going to look for the variational parameter values
SHAPE_RANGE = (0.01, 200)
SCALE_RANGE = (0.001, 100)

# The CAVI algorithm
precision = 10
old_elbo = elbo(data, *ls, precision/2)
print("Initial ELBO:", old_elbo)
print(ls)
while True:
    # Optimize q_1
    l1, l2 = minimize(lambda l: -1*elbo(data, l[0], l[1], ls[2], ls[3], precision), [ls[0], ls[1]],
            bounds=[SHAPE_RANGE, SCALE_RANGE], method="TNC", options={'xtol': 1}).x
    # Optimize q_2
    l3, l4 = minimize(lambda l: -1*elbo(data, l1, l2, l[0], l[1], precision), [ls[2], ls[3]],
            bounds=[SHAPE_RANGE, SCALE_RANGE], method="TNC", options={'xtol': 1}).x
    new_ls = [l1, l2, l3, l4]

    #print("lambdas:", new_ls)
    new_elbo =  elbo(data, *new_ls, precision/2)
    #print("elbo:", new_elbo)
    if np.isclose(old_elbo, new_elbo):
        print(new_ls)
        break
    if new_elbo > old_elbo:
        ls = new_ls
        old_elbo = new_elbo
    else:
        print(ls)
        break

print("Total time:", time() - start_time)
# Compute Approximate KL Divergence
mcmc_points = []
with open("./datasets/full_MCMC_results.csv", mode="r") as f: # Load the MCMC results
    for line in csv.reader(f):
        mcmc_points.append((float(line[0]), float(line[1])))

xs = np.linspace(2, 21, num=36)
ys = np.linspace(2, 90, num=100)

print("Computing KL Divergence")
kl_divergence = 0
for i in range(len(xs)):
    if i == 0:
        continue
    for j in range(len(ys)):
        if j == 0:
            continue
        mcmc_prob = 0.0000000001
        for point in mcmc_points:
            if xs[i-1] < point[0] and point[0] <= xs[i] and ys[j-1] < point[1] and point[1] <= ys[j]:
                mcmc_prob += 1
        mcmc_prob /= len(mcmc_points)

        vi_prob = (gamma.cdf(xs[i], ls[0], scale=ls[1], loc=2) - gamma.cdf(xs[i-1], ls[0], scale=ls[1], loc=2)) * (gamma.cdf(ys[j], ls[2], scale=ls[3], loc=2) - gamma.cdf(ys[j-1], ls[2], scale=ls[3], loc=2))
        if vi_prob == 0:
            continue # D_KL(P || Q) is zero when P is zero
        elif mcmc_prob == 0:
            print(np.inf) # Failure to respect absolute continuity
            exit()

        kl_divergence += vi_prob * np.log(vi_prob/mcmc_prob)

# Note: a negative KL divergence indicates very different supports; KL divergence is likely near np.inf
print(kl_divergence)
