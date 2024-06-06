# # import numpy as np
# # import cvxopt
# # import cvxopt.solvers as solvers
# # import matplotlib.pyplot as plt

# # # 生成数据
# # np.random.seed(1)
# # mean = [0, 0]
# # cov_A = [[1.0, 0], [0, 1.0]]
# # cov_B = [[5.0, 0], [0, 5.0]]
# # A = np.random.multivariate_normal(mean, cov_A, 50)
# # B = np.random.multivariate_normal(mean, cov_B, 50)

# # # SVM
# # X = np.vstack((A, B))
# # y = np.array([1]*50 + [-1]*50).astype('double') # 修改这里

# # m,n = X.shape
# # K = y[:, None] * X
# # K = np.dot(K, K.T)
# # P = cvxopt.matrix(K)
# # q = cvxopt.matrix(-np.ones((m, 1)))
# # G = cvxopt.matrix(np.vstack((-np.eye(m),np.eye(m))))
# # h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * 99999999)))
# # C = cvxopt.matrix(y.reshape(1, -1))
# # b = cvxopt.matrix(np.zeros(1))

# # solvers.options['show_progress'] = False
# # sol = solvers.qp(P, q, G, h, C, b)
# # alphas = np.array(sol['x'])

# # # 获取支持向量
# # S = (alphas > 1e-4).flatten()

# # # 计算 w
# # w = ((y[S] * alphas[S]).T @ X[S]).reshape(-1,1)

# # # 计算预测值
# # y_predict = np.dot(X[S].reshape(-1,1).T, w)
# # # 计算 b
# # b = y[S] - y_predict
# # # 取平均值，因为可能有多个支持向量
# # b = np.mean(b)

# # # 可视化结果
# # fig, ax = plt.subplots()
# # ax.scatter(A[:, 0], A[:, 1], color='blue', label='Class A')
# # ax.scatter(B[:, 0], B[:, 1], color='red', label='Class B')

# # slope = -w[0] / w[1]
# # intercept = -b / w[1]
# # x_vals = np.linspace(-10, 10, 400)
# # y_vals = slope * x_vals + intercept
# # ax.plot(x_vals, y_vals, color='black', linewidth=2)

# # plt.xlabel('x')
# # plt.ylabel('y')
# # plt.title('SVM Classification')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# # # 输出结果
# # print('Alphas: ',alphas[alphas > 1e-4])
# # print('w: ', w.flatten())
# # print('b: ', b[0])
# import numpy as np
# import cvxopt
# import cvxopt.solvers as solvers
# import matplotlib.pyplot as plt

# # 生成数据
# np.random.seed(1)
# mean = [0, 0]
# cov_A = [[1.0, 0], [0, 1.0]]
# cov_B = [[5.0, 0], [0, 5.0]]
# A = np.random.multivariate_normal(mean, cov_A, 50)
# B = np.random.multivariate_normal(mean, cov_B, 50)

# # SVM
# X = np.vstack((A, B))
# y = np.array([1]*50 + [-1]*50).astype('double')

# m,n = X.shape

# # 定义 RBF 核函数
# def rbf_kernel(X, sigma=1.0):
#     m = 100
#     K = np.zeros((m, m))
#     for i in range(m):
#         for j in range(m):
#             K[i,j] = np.exp(-np.linalg.norm(X[i]-X[j])**2 / (2 * (sigma ** 2)))
#     return K

# # 使用 RBF 核函数计算核矩阵
# K = y[:, None] * X
# K = np.dot(K, K.T)
# P = cvxopt.matrix(np.outer(y,y) * rbf_kernel(X))
# q = cvxopt.matrix(-np.ones((m, 1)))
# G = cvxopt.matrix(np.vstack((-np.eye(m),np.eye(m))))
# h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * 99999999)))
# C = cvxopt.matrix(y.reshape(1, -1))
# b = cvxopt.matrix(np.zeros(1))

# solvers.options['show_progress'] = False
# sol = solvers.qp(P, q, G, h, C, b)
# alphas = np.array(sol['x'])

# # 获取支持向量
# S = (alphas > 1e-4).flatten()

# # 计算 RBF 核矩阵
# K = rbf_kernel(X)
# # 计算 b
# b = y[S] - np.sum(alphas[S] * y[S, None] * K[S][:,S], axis=0)
# b = np.mean(b)

# # 可视化结果
# fig, ax = plt.subplots()
# ax.scatter(*A.T, color='blue', label='Class A')
# ax.scatter(*B.T, color='red', label='Class B')

# X1, X2 = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
# X_test = np.array([X1.ravel(), X2.ravel()]).T
# K_test = rbf_kernel(X_test)
# predictions = np.sign(np.sum(alphas * y[:, None] * K_test,axis=0) + b)
# predictions = predictions.reshape(X1.shape)
# plt.contour(X1, X2, predictions, colors='black', levels=[-.5,.5], linestyles=['--','-','--'])

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('SVM Classification with RBF Kernel')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 输出结果
# print('Alphas: ',alphas[alphas > 1e-4])
# print('b: ', b)
import numpy as np
import cvxopt
import cvxopt.solvers as solvers
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(1)
mean = [0, 0]
cov_A = [[1.0, 0], [0, 1.0]]
cov_B = [[5.0, 0], [0, 5.0]]
A = np.random.multivariate_normal(mean, cov_A, 50)
B = np.random.multivariate_normal(mean, cov_B, 50)

# SVM
X = np.vstack((A, B))
y = np.array([1]*50 + [-1]*50).astype('double')

m,n = X.shape

# 定义 RBF 核函数
def rbf_kernel(X1, X2, sigma=3):
    m,n = X1.shape
    K = np.zeros((m,X2.shape[0]))
    for i in range(m):
        for j in range(X2.shape[0]):
            K[i,j] = np.exp(-np.linalg.norm(X1[i]-X2[j])**2 / (2 * (sigma ** 2)))
    return K

# 使用 RBF 核函数计算核矩阵
K = y[:, None] * X
K = np.dot(K, K.T)
P = cvxopt.matrix(np.outer(y,y) * rbf_kernel(X,X))
q = cvxopt.matrix(-np.ones((m, 1)))
G = cvxopt.matrix(np.vstack((-np.eye(m),np.eye(m))))
h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * 100)))
C = cvxopt.matrix(y.reshape(1, -1))
b = cvxopt.matrix(np.zeros(1))

solvers.options['show_progress'] = False
sol = solvers.qp(P, q, G, h, C, b)
alphas = np.array(sol['x'])

# 获取支持向量
S = (alphas > 1e-4).flatten()

# 计算 RBF 核矩阵
K = rbf_kernel(X,X)
# 计算 b
b = y[S] - np.sum(alphas[S] * y[S, None] * K[S][:,S], axis=0)
b = np.mean(b)

# 可视化结果
fig, ax = plt.subplots()
ax.scatter(*A.T, color='blue', label='Class A')
ax.scatter(*B.T, color='red', label='Class B')

X1, X2 = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
X_test = np.array([X1.ravel(), X2.ravel()]).T
K_test = rbf_kernel(X,X_test)
predictions = np.sign(np.sum(alphas[S] * y[S].reshape(-1,1) * K_test[S], axis=0) + b)
predictions = predictions.reshape(X1.shape)
plt.contour(X1, X2, predictions, colors='black', levels=[-.5,.5], linestyles=['--','-','--'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM Classification with RBF Kernel')
plt.legend()
plt.grid(True)
plt.show()

# 输出结果
print('Alphas: ',alphas[alphas > 1e-4])
print('b: ', b)
