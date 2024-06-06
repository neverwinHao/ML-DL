import time
import numpy as np
import matplotlib.pyplot as plt

def tanh(z):
    return np.tanh(z)
def dtanh(z):
    return 1 - z ** 2


class BP:
    def __init__(self, input_layer, hidden_layer, epoch, lr):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.epoch = epoch
        self.lr = lr
        self.weight1 = np.random.uniform(-0.5, 0.5, (input_layer, hidden_layer))
        self.weight2 = np.random.uniform(-0.5, 0.5, (hidden_layer, 1))
        self.bias1 = np.zeros((1, hidden_layer))
        self.bias2 = np.zeros((1, 1))

    def lossfunc(self, y, y_hat):
        loss = np.mean(0.5 * (y - y_hat) ** 2)
        return loss

    def forward(self, X):
        H = np.dot(X.T, self.weight1) + self.bias1
        hidden_out = tanh(H)
        out = np.dot(hidden_out, self.weight2) + self.bias2
        y_hat = out
        return y_hat, hidden_out

    def backward(self, x, y, y_hat, hidden_out):
        pianLy_hat = y_hat - y
        pianLb2 = pianLy_hat
        pianLW2 = np.dot(hidden_out.T, pianLb2)
        pianLb1 = np.dot(pianLb2, self.weight2.T) * dtanh(hidden_out)
        pianLW1 = np.dot(x, pianLb1)

        self.weight1 -= self.lr * pianLW1
        self.weight2 -= self.lr * pianLW2
        self.bias1 -= self.lr * pianLb1
        self.bias2 -= self.lr * pianLb2

    def train(self, inp, out):
        loss = []
        for i in range(self.epoch):
            epoch_loss = 0.0
            for (x, target) in zip(inp, out):
                y_hat, hidden_out = self.forward(x)
                self.backward(x, target, y_hat, hidden_out)
                epoch_loss += self.lossfunc(target, y_hat)

            loss.append(epoch_loss / len(inp))

            if i % 1000 == 0:
                print(f"Epoch: {str(i).ljust(5)} | Loss: {str(loss[-1]).ljust(18)}")

        return loss

    def test(self, inp):
        result = []
        for test_x in inp:
            test_x = test_x.reshape(-1, 1)
            y_hat, _ = self.forward(test_x)
            result.append(y_hat)
        return result


# 生成训练数据
x = np.linspace(-np.pi / 2, np.pi / 2, 100)
train_x = x.reshape(-1, 1)
train_y = np.clip(1 / np.sin(x) + 1 / np.cos(x), -50, 50)

# 初始化神经网络并进行训练
start_time = time.time()

# 定义我的网络
Net = BP(1, 100, epoch=180000, lr=0.005)
losses = Net.train(train_x, train_y)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# 生成测试数据
test_x = np.linspace(-np.pi / 2, np.pi / 2, 200)
test_data = test_x.reshape(-1, 1)

# 进行预测
prediction = Net.test(test_data)

plt.figure(1)
plt.scatter(test_x, prediction, s=4)
plt.xlabel('testX')
plt.ylabel('testY')
plt.show()

plt.figure(2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
