"""
案例2:使用PCA进行图像的降维，k=36
数据集：data/ex7faces.mat
具体算法公式--见笔记第14讲数据降维与重建
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# support functions ---------------------------------------
def plot_n_image(X, n): # 图像可视化,显示n张图片
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


if __name__ == "__main__":
    mat = sio.loadmat('./data/ex7faces.mat')
    X = np.array([x.reshape((32, 32)).T.reshape(1024) for x in mat.get('X')])  # X-->(5000,1024)

    plot_n_image(X, n=64)
    U, _, _ = pca(X)

    # didn't see face in principle components
    plot_n_image(U, n=32)

    # reduce dimension to k=100
    Z = project_data(X, U, k=100)
    plot_n_image(Z, n=64)

    # recover from k=100
    X_recover = recover_data(Z, U) # 选取的Z的前100列(维)数据，还原结果，和原始图像近似
    plot_n_image(X_recover, n=64)
    plt.show()

# -----------------------sklearn PCA-------------------------------------
    from sklearn.decomposition import PCA
    sk_pca = PCA(n_components=100)
    Z = sk_pca.fit_transform(X)
    plot_n_image(Z, 64)
    plt.show()
    X_recover = sk_pca.inverse_transform(Z)
    plot_n_image(X_recover, n=64)
    plt.show()
