"""
案例1：使用PCA进行二维数据的降维
数据集：data/ex7data1.mat
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="white")


# support functions ---------------------------------------
def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                 sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


# PCA functions ---------------------------------------
def covariance_matrix(X):
    """
    Args:
        X (ndarray) (m, n)
    Return:
        cov_mat (ndarray) (n, n):
            covariance matrix of X
    """
    m = X.shape[0]

    return (X.T @ X) / m


def normalize(X):
    """
        for each column, X-mean / std
    """
    X_copy = X.copy()
    m, n = X_copy.shape

    for col in range(n):
        X_copy[:, col] = (X_copy[:, col] - X_copy[:, col].mean()) / X_copy[:, col].std()

    return X_copy


def pca(X):
    """
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    Args:
        X ndarray(m, n)
    Returns:
        U ndarray(n, n): principle components
    """
    # 1. normalize data
    X_norm = normalize(X)

    # 2. calculate covariance matrix
    Sigma = covariance_matrix(X_norm)  # (n, n)

    # 3. do singular value decomposition
    # remeber, we feed cov matrix in SVD, since the cov matrix is symmetry
    # left sigular vector and right singular vector is the same, which means
    # U is V, so we could use either one to do dim reduction
    U, S, V = np.linalg.svd(Sigma)  # U: principle components (n, n)

    return U, S, V


def project_data(X, U, k):
    """
    Args:
        U (ndarray) (n, n)
    Return:
        projected X (n dim) at k dim
    """
    m, n = X.shape

    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]


def recover_data(Z, U):
    m, n = Z.shape

    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')

    return Z @ U[:, :n].T


if __name__ == '__main__':
    mat = sio.loadmat('./data/ex7data1.mat')
    X = mat.get('X')
    X_norm = normalize(X)
    print(X_norm.shape)
    Sigma = covariance_matrix(X_norm)
    print(Sigma)
    U, S, V = pca(X)
    Z = project_data(X_norm, U, 1)  # project data to lower dimension
    print(Z[:10])
    print(Z)
    X_recover = recover_data(Z, U)

    # project data to lower dimension
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    sns.regplot('X1', 'X2',
                data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
                fit_reg=False,
                ax=ax1)
    ax1.set_title('Original dimension')

    sns.rugplot(Z, ax=ax2).set(xlim=(-3, 3))
    ax2.set_xlabel('Z')
    ax2.set_title('Z dimension')

    fig, (ax3, ax4, ax5) = plt.subplots(ncols=3, figsize=(16, 4))

    # recover data to original dimension
    sns.rugplot(Z, ax=ax3).set(xlim=(-3, 3))
    ax3.set_title('Z dimension')
    ax3.set_xlabel('Z')

    sns.regplot('X1', 'X2',
                data=pd.DataFrame(X_recover, columns=['X1', 'X2']),
                fit_reg=False,
                ax=ax4)
    ax4.set_title("2D projection from Z")

    sns.regplot('X1', 'X2',
                data=pd.DataFrame(X_norm, columns=['X1', 'X2']),
                fit_reg=False,
                ax=ax5)
    ax5.set_title('Original dimension')
    plt.show()
