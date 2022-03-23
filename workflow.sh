#!/bin/bash

# All code was originally run on Python 3.6.9
# It appears that the seed() function from this version is deprecated/legacy
# Keep this in mind when using newer versions
python3 --version
echo

# Code for CAVI on a normal sample with closed form updates
time python3 normal_dist_analytic.py 10 1 10 1
echo

time python3 normal_dist_analytic.py 25 1 10 1
echo

time python3 normal_dist_analytic.py 10 1 20 1
echo

# Code for CAVI on a normal with numerical optimization of the ELBO
time python3 normal_dist_cavi.py 10 1 10 1
echo

time python3 normal_dist_cavi.py 25 1 10 1
echo

time python3 normal_dist_cavi.py 10 1 20 1
echo

# Code for MCMC on normal
time python3 normal_dist_hybrid.py 10 10 20000 5000
echo

time python3 normal_dist_hybrid.py 25 10 20000 5000
echo

time python3 normal_dist_hybrid.py 10 20 20000 5000
echo

# Code for Hybrid CAVI on normal
time python3 normal_dist_hybrid.py 10 10 1000 100 hybrid
echo

time python3 normal_dist_hybrid.py 25 10 1000 100 hybrid
echo

time python3 normal_dist_hybrid.py 10 20 1000 100 hybrid
echo

# All code for the t distribution
python3 t_dist_hybrid.py MCMC
echo

python3 t_dist_hybrid.py Hybrid 4 4
echo

python3 t_dist_hybrid.py Hybrid 30 30
echo

python3 t_dist_hybrid.py CAVI 10 10 10 10
echo

python3 t_dist_hybrid.py CAVI 3.2 0.6 1.7 6.9
echo

python3 t_dist_hybrid.py CAVI 1.7 6.9 3.2 0.6
echo
