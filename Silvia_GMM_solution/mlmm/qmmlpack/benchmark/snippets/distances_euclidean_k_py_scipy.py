# qmmlpack
# (c) Matthias Rupp, 2006-2016.
# See LICENSE.txt for license.

import scipy.spatial

description = "Euclidean distance matrix K, Python/SciPy";

ngrid = [10, 50, 100, 500, 1000, 5000]    # number of training data
dgrid = [1, 10, 50, 100, 300, 500, 1000]  # dimensionality of training data
theta = [['n', ngrid], ['d', dgrid]]      # grid values are automatically set up

def before(thetaval, thetaind):
    global xx
    xx = (np.random.rand(thetaval[0], thetaval[1]) - 0.5) * 200

def after(thetaval, thetaind, result):
    global xx
    if not result.shape == (thetaval[0], thetaval[0]): bm_error("non-matrix result for theta = {}.".format(thetaval))
    del xx

def distance_euclidean(xx): return sp.spatial.distance.squareform(sp.spatial.distance.pdist(xx, 'euclidean'))
def function(thetaval, thetaind): return distance_euclidean(xx)
