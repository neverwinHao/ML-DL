import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

def data():
    # 设置均值和协方差矩阵
    mean = np.array([0, 0])
    num_samples = 50
    cov_matrixA = np.array([[1, 0], [0, 1]])
    cov_matrixB = np.array([[5, 0], [0, 5]])
    # 生成随机数据点
    np.random.seed(0)
    random_pointA = np.random.multivariate_normal(mean, cov_matrixA, num_samples)
    label_A = -np.ones(50)  # A类标签为0
    random_pointB = np.random.multivariate_normal(mean, cov_matrixB, num_samples)
    label_B = np.ones(50)  # B类标签为 1
    # 画原始点分布
    # plt.scatter(random_pointA[:, 0], random_pointA[:, 1], c='red', label='Class A')
    # plt.scatter(random_pointB[:, 0], random_pointB[:, 1], c='blue', label='Class B')
    # plt.title("Origin Point")
    # plt.legend(loc='best')
    # plt.show()
    return random_pointA, label_A, random_pointB, label_B


def plot_decision_boundary(clf, X, Y):
    # 生成网格点来绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # 使用分类器预测每个点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和训练数据点
    plt.contourf(xx, yy, Z, alpha=0.4, levels=np.linspace(-1, 1, 3), cmap='inferno')
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    # class_A = plt.scatter([], [], c='red', label='Class A')
    # class_B = plt.scatter([], [], c='blue', label='Class B')
    # plt.legend(handles=[class_A, class_B], loc='best')
    plt.scatter(x_A[:, 0], x_A[:, 1], c='red', label='Class A')
    plt.scatter(x_B[:, 0], x_B[:, 1], c='blue', label='Class B')
    plt.legend(loc='best')
    plt.title(f"C:{1} | Acc:{86.0}")
    plt.show()


x_A, y_A, x_B, y_B = data()

X = np.vstack((x_A, x_B))
Y = np.hstack((y_A, y_B))

# 建立 svm 模型
start = time.time()
clf = svm.SVC(kernel='rbf', random_state=0, gamma=1, C=1)
clf.fit(X, Y)
y_pred = clf.predict(X)
plot_decision_boundary(clf, X, Y)
end = time.time()
accuracy = accuracy_score(Y, y_pred)
print(end-start)
print("Accuracy:", accuracy)
