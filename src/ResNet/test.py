#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import os
import matplotlib.pyplot as plt
from Experi3_ResNet import ResNet



# 指定当前的GPU设备
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

# # 加载已经训练好的模型
model = ResNet.ResNet18(num_class=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.load_state_dict(torch.load("models/Mymodel.pth"))
model.to(device)
model.eval()


dataloader_train, dataloader_test = ResNet.read_data(batch_size=256)
# 定义测试函数
def test(model, test_num, test_loader):
    model.eval()
    count = 0  # 记录已处理的样本数量
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    output_folder = 'Test'
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            if count >= test_num:
                break

            batch_size = X.size(0)  # 当前批次中的样本数量
            for j in range(batch_size):
                img_true, label = X[j][0].numpy(), y[j].item()
                X_batch = X.to(device)

                pred = model(X_batch)
                y_pred = torch.argmax(pred, dim=1)[j].item()

                true_label = classes[label]
                predicted_label = classes[y_pred]

                print("-----------验证模型性能开始------------")
                print("预测结果：", true_label)
                print("真实标签：", predicted_label)

                # 显示图像
                plt.imshow(img_true)
                plt.title(f"True Label: {true_label}, Predicted Label: {predicted_label}")
                plt.savefig(os.path.join(output_folder, f"预测结果{count}.png"))
                # plt.show()

                count += 1
                if count >= test_num:
                    break

if __name__ == '__main__':
    # 调用测试函数
    test(model, 6, dataloader_test)


