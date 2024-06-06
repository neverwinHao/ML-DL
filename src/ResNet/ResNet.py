import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import time

# 指定当前的GPU设备
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


# 搭建block部分
class Block(nn.Module):
    def __init__(self, inp_channel, out_channel, stride=1):
        super(Block, self).__init__()
        # 此处bias设false是为了避免和后面BN层的bias冲突
        self.conv1 = nn.Conv2d(inp_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        # 维度对齐
        if stride != 1 or inp_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(identity)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(2, 64, 64, stride=1)

        self.layer2 = self.make_layer(2, 64, 128, stride=2)

        self.layer3 = self.make_layer(2, 128, 256, stride=2)
        self.layer4 = self.make_layer(2, 256, 512, stride=2)

        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_class)

    # 此函数用来方便增加层数
    def make_layer(self, num_block, inp_channel, out_channel, stride):
        net = []
        net.append(Block(inp_channel, out_channel, stride))

        for u in range(1, num_block):
            net.append(Block(out_channel, out_channel, stride))
        net = nn.Sequential(*net)
        return net

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpooling(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out


def read_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_train = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )
    data_test = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader_train, dataloader_test


dataloader_train, dataloader_test = read_data(batch_size=256)
num_class = 10
model = ResNet18(num_class)

# 由于ResNet是针对ImageNet的，而CIFAR-10数据集图片较小，修改网络结构，保留更多信息
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

model = model.to(device)
# 损失函数
lossfunc = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(model, lossfunc, optimizer, dataloader_train, dataloader_test):
    train_loss = 0.0
    train_acc = 0.0
    count = 0
    start = time.time()

    for i, (X, y) in enumerate(dataloader_train):
        optimizer.zero_grad()
        # 每次提取到 X，y 都先转到 GPU
        (X, y) = X.to(device), y.to(device)

        # 计算当前批次的输出
        y_hat = model(X)

        # # 计算当前批次的损失值
        loss_batch = lossfunc(y_hat, y)

        # 计算预测值
        _, y_pred = torch.max(y_hat, axis=1)

        # 计算当前批次的准确率
        acc_batch = torch.sum(y_pred == y) / y_hat.shape[0]


        loss_batch.backward()
        optimizer.step()

        train_loss += loss_batch.item()
        train_acc += acc_batch.item()
        count += 1

    end = time.time()
    train_time = end - start
    train_loss = train_loss / count
    train_acc = train_acc * 100 / count

    # 计算在测试集上的精度
    model.eval()
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader_test):
            (X, y) = X.to(device), y.to(device)

            y_hat = model(X)
            _, y_pred = torch.max(y_hat, axis=1)

            correct_predictions += torch.sum(y_pred == y)
            total_samples += y.shape[0]

    test_accuracy = correct_predictions.item() / total_samples * 100
    # print("测试精度：", test_accuracy)
    # print(f"Train Loss: {train_loss} | Train Acc: {train_acc}% | Test Acc:{test_accuracy}% | Train Time:{end-start}")
    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} % | Test Acc: {test_accuracy:.3f} % | Train Time: {train_time:.3f}")

    return train_loss, test_accuracy


if __name__ == '__main__':
    # 开始训练
    epoch = 200
    min_acc = 0
    loss_list = []
    for i in range(epoch):
        print(f'第{i + 1}轮训练\n----------------------------')
        train_loss, test_accuracy = train(model, lossfunc, optimizer, dataloader_train, dataloader_test)
        loss_list.append(train_loss)

        # 保存最优模型
        if test_accuracy > min_acc:
            min_acc = test_accuracy
            torch.save(model.state_dict(), "models/Mymodel.pth")
            best_model = i+1
            print('最佳模型保存成功！')
        print(f"最佳模型第{best_model}轮产生 | 最佳测试精度为{min_acc:.3f} %")
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print("最优模型准确率：", min_acc)
    print("训练完成!")
