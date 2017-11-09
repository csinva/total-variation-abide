import os, sys, time, subprocess, h5py, argparse, logging, pickle, random
import numpy as np
from os.path import join as oj
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets
from nilearn import plotting
import cvxpy as cvx
import cvxopt as cvxopt

X = np.loadtxt('data/Caltech_0051461_rois_dosenbach160.1D')[:, :-1]
print(X.shape)


# plot covariance matrix
def plot_cov():
    covs = np.cov(X.transpose())
    # plt.imshow(covs)
    sns.clustermap(covs)
    plt.savefig('figs/covs.pdf')


def plot_idxs(idxs_list):
    for idx in idxs_list:
        plt.plot(X[:, idx])
    plt.savefig('figs/time_course.pdf')


def plot_connectome():
    g = np.zeros(shape=(160, 160))

    dos_coords = datasets.fetch_coords_dosenbach_2010()
    dos_coords = dos_coords.rois
    dos_coords_table = [[x, y, z] for (x, y, z) in dos_coords]  # Reformat the atlas coordinates

    f = plt.figure(figsize=(2.3, 3.5))  # 2.2,2.3
    plotting.plot_connectome(g, dos_coords_table, display_mode='z',
                             output_file='figs/connectome.pdf',
                             annotate=False, figure=f, node_size=18)
    # plt.show()


# plot_cov()
# plot_idxs([58, 139])
# plot_connectome()

y = X[:, 58]

# Set regularization parameter.
vlambda = 50
# Solve l1 trend filtering problem.
x = cvx.Variable(y.size)
obj = cvx.Minimize(0.5 * cvx.sum_squares(y - x)
                    + vlambda * cvx.tv(x))
                   # + vlambda * cvx.norm(x, 1))
prob = cvx.Problem(obj)
# ECOS and SCS solvers fail to converge before
# the iteration limit. Use CVXOPT instead.
prob.solve(solver=cvx.CVXOPT, verbose=True)

# print('Solver status: ', prob.status)
# Check for error.
if prob.status != cvx.OPTIMAL:
    raise Exception("Solver did not converge!")

plt.plot(y, label='original')
plt.plot(x.value, label='TV regularized')
plt.legend()
plt.show()
