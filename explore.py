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
def plot_cov(covs, tv=False):
    sns.clustermap(covs)
    if tv:
        plt.savefig('figs/covs_tv.pdf')
    else:
        plt.savefig('figs/covs.pdf')


def plot_idxs(idxs_list):
    for idx in idxs_list:
        plt.plot([2 * i for i in range(X[:, idx].size)], X[:, idx])
    plt.xlabel('Time (s)')
    plt.ylabel('Neural response')
    plt.savefig('figs/time_course.pdf')


def plot_connectome(covs, tv=False):
    # g = np.zeros(shape=(160, 160))
    g = covs
    dos_coords = datasets.fetch_coords_dosenbach_2010()
    dos_coords = dos_coords.rois
    dos_coords_table = [[x, y, z] for (x, y, z) in dos_coords]  # Reformat the atlas coordinates

    f = plt.figure(figsize=(2.3, 3.5))  # 2.2,2.3
    if tv:
        plotting.plot_connectome(g, dos_coords_table, display_mode='z',
                                 output_file='figs/connectome_tv.pdf', edge_threshold="0.0005%",
                                 annotate=True, figure=f, node_size=18)

    else:
        plotting.plot_connectome(g, dos_coords_table, display_mode='z',
                                 output_file='figs/connectome.pdf', edge_threshold="0.0005%",
                                 annotate=True, figure=f, node_size=18)


def plot_tv_vary_lambda():
    y = X[:, 58]
    plt.plot([2 * i for i in range(y.size)], y, label='original')
    lambdas = [1, 10, 25]
    for vlambda in lambdas:
        # vlambda = 50

        x = cvx.Variable(y.size)
        obj = cvx.Minimize(0.5 * cvx.sum_squares(y - x)
                           + vlambda * cvx.tv(x))
        prob = cvx.Problem(obj)
        # ECOS and SCS solvers fail to converge before
        # the iteration limit. Use CVXOPT instead.
        prob.solve(solver=cvx.CVXOPT, verbose=True)
        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")

        plt.plot([2 * i for i in range(y.size)], x.value, label='TV  $\lambda=$' + str(vlambda))
    plt.legend()
    plt.xlim([0, 150])
    plt.xlabel('Time (s)')
    plt.ylabel('Neural response')
    plt.savefig('figs/tv_vary.pdf')
    plt.show()


def plot_tv_vary_lambda_diff():
    y = X[:, 58]
    plt.plot([2 * i for i in range(np.diff(y).size)], np.diff(y), label='original')
    lambdas = [1, 10, 25]
    for vlambda in lambdas:
        # vlambda = 50

        x = cvx.Variable(y.size)
        obj = cvx.Minimize(0.5 * cvx.sum_squares(y - x)
                           + vlambda * cvx.tv(x))
        prob = cvx.Problem(obj)
        # ECOS and SCS solvers fail to converge before
        # the iteration limit. Use CVXOPT instead.
        prob.solve(solver=cvx.CVXOPT, verbose=True)
        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")
        xv = np.array(x.value).flatten()
        plt.plot([2 * i for i in range(np.diff(xv).size)], np.diff(xv), label='TV  $\lambda=$' + str(vlambda))

    plt.legend()
    plt.xlim([0, 150])
    plt.xlabel('Time (s)')
    plt.ylabel('Neural response 1st derivative')
    plt.savefig('figs/tv_vary_diff.pdf')
    plt.show()


def tv_X(X):
    X_out = np.zeros(shape=(146, 160))
    for i in range(160):
        print(i)
        Y = X[:, i]
        vlambda = 10

        x = cvx.Variable(Y.size)
        obj = cvx.Minimize(0.5 * cvx.sum_squares(Y - x)
                           + vlambda * cvx.tv(x))
        prob = cvx.Problem(obj)
        # ECOS and SCS solvers fail to converge before
        # the iteration limit. Use CVXOPT instead.
        prob.solve(solver=cvx.CVXOPT, verbose=False)
        if prob.status != cvx.OPTIMAL:
            raise Exception("Solver did not converge!")
        X_out[:, i] = np.array(x.value).ravel()
    return X_out


def tv_full(X):
    Y = X[:, :10]
    vlambda = 10

    x = cvx.Variable(Y.shape[0], 10)
    obj = cvx.Minimize(0.5 * cvx.sum_squares(Y - x)
                       + vlambda * cvx.tv(x))
    prob = cvx.Problem(obj)
    # ECOS and SCS solvers fail to converge before
    # the iteration limit. Use CVXOPT instead.
    prob.solve(solver=cvx.CVXOPT, verbose=False)
    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")

    return x.value

# x2 = tv_full(X)
# pickle.dump(x2, open('x2.pkl', 'wb'))

# covs = np.cov(X.transpose())

# plot_cov(covs)
# print('num nonzero', np.sum(covs != 0))
# cflat = covs.flatten()
# ind = np.argpartition(cflat, -10)[-10:]
# print(ind)
# cflat[ind] = -1
# cflat[cflat != -1] = 0
# covs = cflat.reshape(160, 160)
# for r in range(160):
#     for c in range(160):
#         if covs[r, c] != 0:
#             covs[c, r] = -1
# print(np.sum(covs))
# plot_connectome(covs)

# x = tv_X(X)
# pickle.dump(x, open('x.pkl', 'wb'))
# print(x.shape)
# x = pickle.load(open('x.pkl', 'rb'))
# covs = np.cov(x.transpose())
# plt.plot(x[:, 58])
# plt.plot(X[:, 58])
# plt.show()
# plot_cov(covs, tv=True)
# plot_connectome(covs, tv=True)



# individual plots
# plot_idxs([58, 139])
# plot_tv_vary_lambda()
plot_tv_vary_lambda_diff()
# plt.show()
