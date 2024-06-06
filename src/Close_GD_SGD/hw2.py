import numpy as np
import matplotlib.pyplot as plt

# 设置均值和协方差矩阵
mean = np.array([2, 1])
cov_matrix = np.array([[1, -0.5], [-0.5, 2]])
num_samples = 30
# 生成随机数据点,保证每次随机得到的结果相同
np.random.seed(0)

random_points = np.random.multivariate_normal(mean, cov_matrix, num_samples)

# 提取x和y坐标
x_val = random_points[:, 0]
y_val = random_points[:, 1]

# 构造矩阵
X_origin = np.array(x_val)
bias = np.ones(X_origin.shape)

# 增加偏置后的矩阵
X = np.column_stack((bias, x_val))
Y = np.array(y_val)

final_theta = []


# # 绘制随机点的散点图
# plt.scatter(x_val, y_val)
# plt.title('Random Points')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.show()

# 闭式解算法
def close_form():
    # 计算闭式解

    # X_1 = X的转置*X-->逆
    X_1 = np.linalg.pinv(np.dot(X.T, X))
    theta = np.dot(np.dot(X_1, X.T), Y)
    y_hat = theta[0] + x_val * theta[1]

    final_theta.append(theta)
    # 绘制原始数据点
    plt.scatter(x_val, y_val, label='Random Points')

    # 绘制预测线
    plt.plot(x_val, y_hat, color='red', label='Prediction')
    plt.title('close form solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


# J的梯度
def gradJ(X, Y, theta):
    temp = -(Y - np.dot(X, theta))
    grad = np.dot(X.T, temp)
    return grad / len(X)


# GD Solution
def GD():
    lr = 0.1
    theta = np.random.rand(2)
    epoch = 600
    batch = 30
    theta_list = []

    # 初始直线
    y_origin = theta[0] + x_val * theta[1]

    # 开始训练
    for i in range(epoch):
        theta = theta - lr * gradJ(X, Y, theta)
        grad_norm = np.linalg.norm(gradJ(X, Y, theta), ord=2)
        if i % batch == 0:
            theta_list.append(theta)
        if grad_norm < 0.001:
            break
    final_theta.append(theta)
    plt.scatter(x_val, y_val, label='Random Points')

    for i, theta_i in enumerate(theta_list):
        if i == len(theta_list) - 1:
            plt.plot(x_val, theta_i[0] + x_val * theta_i[1], color='red', label='Final Prediction', linewidth=2)
        else:
            plt.plot(x_val, theta_i[0] + x_val * theta_i[1], color='black', alpha=0.5, linestyle='--', linewidth=1)

    plt.plot(x_val, y_origin, color='blue', label='Origin Prediction')
    plt.title('GD solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


def single_grad(x, y, theta):
    grad = np.zeros(2)
    grad[0] = (theta[0] + theta[1] * x - y)
    grad[1] = x * (theta[0] + theta[1] * x - y)
    return grad


def SGD():
    lr = 0.001
    theta = np.random.rand(2)
    epoch = 2000
    y_origin = theta[0] + x_val * theta[1]
    # 开始训练
    for i in range(epoch):
        for (x, y) in zip(x_val, y_val):
            theta = theta - lr * single_grad(x, y, theta)
    y_hat = theta[0] + x_val * theta[1]
    plt.scatter(x_val, y_val, label='Random Points')
    plt.plot(x_val, y_origin, color='blue', label='Origin Prediction')
    plt.plot(x_val, y_hat, color='red', label='Final Prediction')
    plt.title('SGD solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    close_form()
    GD()
    SGD()
# if __name__ == "__main__":
#     close_form()
#     GD()
#     SGD()
#     for i,theta in enumerate(final_theta):
#         y_hat = theta[0]+ x_val*theta[1]
#         if i == 0:
#             plt.plot(x_val, y_hat, color='red', label='close')
#         elif i==1:
#             plt.plot(x_val, y_hat, color='blue', label='GD')
#         elif i==2:
#             plt.plot(x_val, y_hat, color='black', label='SGD')
#     plt.scatter(x_val, y_val, label='Random Points')
#     # plt.title('三种方法对比')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
