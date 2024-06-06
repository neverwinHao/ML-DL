import numpy as np
import matplotlib.pyplot as plt


# 由于数据不变，此处采用hw3的代码
def data():
    # 设置均值和协方差矩阵
    mean = np.array([0, 0])
    num_samples = 50
    cov_matrixA = np.array([[1, 0], [0, 1]])
    cov_matrixB = np.array([[5, 0], [0, 5]])
    # 生成随机数据点
    np.random.seed(0)
    random_pointA = np.random.multivariate_normal(mean, cov_matrixA, num_samples)
    label_A = -np.ones(num_samples)
    random_pointB = np.random.multivariate_normal(mean, cov_matrixB, num_samples)
    label_B = np.ones(num_samples)
    # 画原始点分布
    plt.scatter(random_pointA[:, 0], random_pointA[:, 1], c='red', label='Class A')
    plt.scatter(random_pointB[:, 0], random_pointB[:, 1], c='blue', label='Class B')
    plt.legend(loc='best')
    # plt.show()
    return random_pointA, label_A, random_pointB, label_B


x_A, y_A, x_B, y_B = data()

X = np.vstack((x_A, x_B))
Y = np.hstack((y_A, y_B))


def Gauss_kernel(x, y):
    sigma = 1.0
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


n_samples = X.shape[0]
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i, j] = Gauss_kernel(X[i], X[j])
