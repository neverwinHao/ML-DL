import numpy as np
import matplotlib.pyplot as plt


def Gauss_data():
    # 设置均值和协方差矩阵
    meanA = np.array([0, 0])
    meanB = np.array([2, 0])
    meanC = np.array([0, 2])
    cov_matrixA = np.array([[1, 0], [0, 1]])
    cov_matrixB = np.array([[0.5, 0], [0, 1]])
    cov_matrixC = np.array([[1, 0.3], [0.3, 1]])
    num_samples = 25

    # 生成随机数据点
    np.random.seed(10)
    random_pointA = np.random.multivariate_normal(meanA, cov_matrixA, num_samples)
    random_pointB = np.random.multivariate_normal(meanB, cov_matrixB, num_samples)
    random_pointC = np.random.multivariate_normal(meanC, cov_matrixC, num_samples)
    return random_pointA, random_pointB, random_pointC


random_pointA, random_pointB, random_pointC = Gauss_data()
dataset = np.vstack((random_pointA, random_pointB, random_pointC))
k = 3


class kmeans:
    # 数据集和分成k类
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.num_data = len(self.dataset)

    def Random_init(self):
        # 随机选择k个初始点
        indices = np.random.choice(self.num_data, self.k, replace=False)
        initial_points = self.dataset[indices]
        return initial_points

    def furtherst_init(self):
        indices = np.random.choice(self.num_data, 1, replace=False)
        initial_points = self.dataset[indices]
        initial_points = np.reshape(initial_points, (1, -1))

        for _ in range(1, self.k):
            distances = np.zeros(self.num_data)
            for i in range(self.num_data):
                point_distances = np.linalg.norm(self.dataset[i] - initial_points, axis=1)
                distances[i] = np.min(point_distances)

            farthest_point_index = np.argmax(distances)
            farthest_point = self.dataset[farthest_point_index]

            initial_points = np.vstack((initial_points, farthest_point))

        return initial_points

    def kmeans_add_init(self):
        # 随机选择第一个初始点
        init_index = np.random.choice(self.num_data, 1, replace=False)
        init_point = self.dataset[init_index]

        for _ in range(1, self.k):
            distances = np.zeros(self.num_data)
            for i in range(self.num_data):
                point_distances = np.linalg.norm(self.dataset[i] - init_point, axis=1)
                distances[i] = np.min(point_distances)

            prob = distances / np.sum(distances)  # 计算每个样本被选择为新的初始点的概率
            next_index = np.random.choice(self.num_data, 1, p=prob)  # 根据概率随机选择一个样本
            next_point = self.dataset[next_index]

            initial_points = np.vstack((init_point, next_point))

        return initial_points

    # 寻找距离其他中心点最短的距离
    def find_close(self, center):
        close_id = np.zeros((self.num_data, 1))
        for data_index in range(self.num_data):
            distances = np.linalg.norm(self.dataset[data_index] - center, axis=1)
            closest_center_index = np.argmin(distances)
            close_id[data_index] = closest_center_index
        return close_id

    # 更新中心点的位置
    def update(self, close_id):
        center_update = np.zeros((self.k, self.dataset.shape[1]))  # 初始化更新后的中心点

        for i in range(self.k):
            # 找到属于第i个簇的数据点的索引
            cluster_indices = np.where(close_id == i)[0]

            if len(cluster_indices) > 0:
                # 计算该簇内所有数据点的均值作为新的中心点位置
                center_update[i] = np.mean(self.dataset[cluster_indices], axis=0)  # axis = 0表示按列求平均值

        return center_update

    def train(self, epoch):
        # # Random方法选择初始点
        # center = self.Random_init()

        # # 最远法选择初始点
        # center = self.furtherst_init()
        # k_means++方法找初始点
        center = self.kmeans_add_init()
        # 找这七十五个点到哪个中心点距离最近，初始矩阵75*1
        for _ in range(epoch):
            close_id = self.find_close(center)
            # 更新中心点的位置
            center = self.update(close_id)

        return center, close_id


MyKmeans = kmeans(dataset, k)
center, close_id = MyKmeans.train(epoch=1000)


# # 画原始点分布
# plt.figure(figsize=(12, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Origin')
# plt.scatter(random_pointA[:, 0], random_pointA[:, 1], c='red', label='Class A')
# plt.scatter(random_pointB[:, 0], random_pointB[:, 1], c='blue', label='Class B')
# plt.scatter(random_pointC[:, 0], random_pointC[:, 1], c='green', label='Class C')
# plt.legend(loc='best')
#
# # 可视化分类结果
# colors = ['red', 'blue', 'green']
#
# plt.subplot(1, 2, 2)
# # plt.title('Radom Init Kmeans')
# # plt.title('Farthest Init Kmeans')
# plt.title('Kmeans++')
# for i in range(k):
#     cluster_indices = np.where(close_id == i)[0]
#     plt.scatter(dataset[cluster_indices, 0], dataset[cluster_indices, 1], c=colors[i], label='Class {}'.format(i + 1))
#
# plt.scatter(center[:, 0], center[:, 1], c='black', marker='x', label='Centroids')
# plt.legend(loc='best')
# plt.show()


def calculate_sse(dataset, center, close_id):
    sse = 0
    for i in range(center.shape[0]):
        cluster_indices = np.where(close_id == i)[0]
        sse += np.sum(np.square(dataset[cluster_indices] - center[i]))
    return sse


def choose_k(dataset, max_k):
    k_values = []
    sse_values = []

    for k in range(2, max_k + 1):
        MyKmeans = kmeans(dataset, k)
        center, close_id = MyKmeans.train(epoch=1000)
        sse = calculate_sse(dataset, center, close_id)

        k_values.append(k)
        sse_values.append(sse)

    plt.plot(k_values, sse_values, marker='o')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.title('Choose K')
    plt.show()


choose_k(dataset, max_k=13)
