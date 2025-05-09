# qmmlpack
# (c) Matthias Rupp, 2006-2016.
# See LICENSE.txt for license.

import scipy.linalg

description = "Forward substitution (vector), Python/C++/BLAS2";

ngrid = [10, 50, 100, 500, 1000, 5000, 10000];  # size of matrix
theta = [["n", ngrid]];  # thetaGrid computed automatically

def before(thetaval, thetaind):
    global xx, b
    xx = np.tril((np.random.rand(thetaval[0], thetaval[0]) - 0.5) * 200)
    b  = (np.random.rand(thetaval[0]) - 0.5) * 200

def after(thetaval, thetaind, result):
    global xx, b
    if not result.shape == (thetaval[0],): bm_error("non-vector result for theta = {}.".format(thetaval))
    del xx, b

def function(thetaval, thetaind): return qmml.forward_substitution(xx, b)
