import numpy as np
import matplotlib.pyplot as plt

# 生成标准的点，便于与吉布斯采样后对比是否正确

mean = np.array([0, 0])
a = 100
b = 99
cov_matrix = np.array([[100, 99], [99, 100]])
standard_point = np.random.multivariate_normal(mean, cov_matrix, 50)

# 迭代次数
iterations = 100
x_list = []
y_list = []
for _ in range(50):
    x = np.random.randn()
    y = np.random.randn()
    for _ in range(iterations):
        x = np.random.normal(b * y / a, a - b ** 2 / a)
        y = np.random.normal(b * x / a, a - b ** 2 / a)
    x_list.append(x)
    y_list.append(y)

# plt.figure(figsize=(8, 6))
# plt.scatter(x_list, y_list, c='red', label='Gibbs Point')
# plt.scatter(standard_point[:, 0], standard_point[:, 1], c='blue', label='Standard Point', alpha=0.5)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Comparison of Gibbs Point and Standard Point')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# 绘制吉布斯采样点的图
axs[0].scatter(x_list, y_list, c='red', label='Gibbs Point')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Gibbs Point')

# 绘制标准点的图
axs[1].scatter(standard_point[:, 0], standard_point[:, 1], c='blue', label='Standard Point', alpha=0.5)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('Standard Point')

# 显示图形
for ax in axs.flat:
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
