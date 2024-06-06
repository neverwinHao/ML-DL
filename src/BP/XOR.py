import time
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dsigmoid(z):
    return z * (1 - z)


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
        hidden_out = sigmoid(H)
        out = np.dot(hidden_out, self.weight2) + self.bias2
        y_hat = sigmoid(out)
        return y_hat, hidden_out

    def backword(self, x, y, y_hat, hidden_out):
        pianLy_hat = y_hat - y
        pianLb2 = pianLy_hat * dsigmoid(y_hat)
        pianLW2 = np.dot(hidden_out.T, pianLb2)
        pianLb1 = np.dot(pianLb2, self.weight2.T) * dsigmoid(hidden_out)
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
                x = x.reshape(-1, 1)
                y_hat, hidden_out = self.forward(x)
                self.backword(x, target, y_hat, hidden_out)
                epoch_loss += self.lossfunc(target, y_hat)

            loss.append(epoch_loss / len(inp))

            if i % 1000 == 0:
                print(f"Epoch: {str(i).ljust(5)} | Loss: {str(loss[-1]).ljust(18)}")

        return loss


    def test(self, inp):
        result = []
        for x in inp:
            x = x.reshape(-1, 1)
            y_hat, hidden_out = self.forward(x)
            result.append(y_hat)
        return np.round(result)



# 定义训练数据和测试数据
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
train_y = np.array([[0], [1], [1], [0]])
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

start_time = time.time()
Net = BP(2, 4, epoch=15000, lr=0.1)
losses = Net.train(train_x, train_y)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time: {training_time:.2f} seconds")

# 修改部分连接权值
# Net.weight1[0][1] = 0
# Net.weight1[0][2] = 0

prediction = Net.test(test_data)
print("Predictions:")
print(prediction)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
