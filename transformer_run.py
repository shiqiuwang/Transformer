import torch.optim as optim
import torch.utils.data as data
from generate_url_datasets import url_train_data, url_valid_data
from my_dataset import MyDataset
import argparse
from torch.utils.data import DataLoader
import random

from model import *
from config import Config


from torch import nn
import torch

torch.cuda.set_device(0)
torch.manual_seed(2020)
EPOCH = 20

train_loss_curve = list()
valid_loss_curve = list()

train_acc_curve = list()
valid_acc_curve = list()

train_acc_curve0 = []
train_acc_curve1 = []
train_acc_curve2 = []
train_acc_curve3 = []
train_acc_curve4 = []
train_acc_curve5 = []

valid_acc_curve0 = []
valid_acc_curve1 = []
valid_acc_curve2 = []
valid_acc_curve3 = []
valid_acc_curve4 = []
valid_acc_curve5 = []

train_real_label = []
train_pre_label = []

valid_real_label = []
valid_pre_label = []
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--label_num', type=int, default=6)
parser.add_argument('--seed', type=int, default=2020)
args = parser.parse_args()

# 设置随机种子
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)

device = torch.device("cuda:0")

training_set = MyDataset(url_train_data)

train_loader = data.DataLoader(dataset=training_set,
                               batch_size=128, shuffle=True)

valid_data = MyDataset(url_valid_data)
valid_loader = DataLoader(dataset=valid_data, batch_size=128)

config = Config()
model = Transformer(config)

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Train the model
for epoch in range(20):
    loss_mean = 0
    correct = 0
    total = 0

    total0 = 0
    total1 = 0
    total2 = 0
    total3 = 0
    total4 = 0
    total5 = 0

    correct0 = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0
    correct5 = 0

    model.train()

    for i, data in enumerate(train_loader):
        deep_data, target = data[0][:, 34:-1], data[0][:, -1]
        deep_data = deep_data.to(device)
        target = target.to(device)
        deep_data = deep_data.view(128, 1, 39, 100)
        outputs = model(deep_data)

        optimizer.zero_grad()
        loss = criterion(outputs, target.long())
        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        for target_val in target.long().cpu().numpy():
            train_real_label.append(target_val)
        for pre_val in predicted.cpu().numpy():
            train_pre_label.append(pre_val)
        total += target.size(0)
        correct += (predicted == target.long()).cpu().squeeze().sum().numpy()

        type_index0 = np.where(target.cpu().numpy() == 0.0)
        total0 += len(type_index0[0])
        for index in type_index0[0]:
            if predicted[index] == target.long()[index]:
                correct0 += 1

        type_index1 = np.where(target.cpu().numpy() == 1.0)
        total1 += len(type_index1[0])
        for index in type_index1[0]:
            if predicted[index] == target.long()[index]:
                correct1 += 1

        type_index2 = np.where(target.cpu().numpy() == 2.0)
        total2 += len(type_index2[0])
        for index in type_index2[0]:
            if predicted[index] == target.long()[index]:
                correct2 += 1

        type_index3 = np.where(target.cpu().numpy() == 3.0)
        total3 += len(type_index3[0])
        for index in type_index3[0]:
            if predicted[index] == target.long()[index]:
                correct3 += 1

        type_index4 = np.where(target.cpu().numpy() == 4.0)
        total4 += len(type_index4[0])
        for index in type_index4[0]:
            if predicted[index] == target.long()[index]:
                correct4 += 1

        type_index5 = np.where(target.cpu().numpy() == 5.0)
        total5 += len(type_index5[0])
        for index in type_index5[0]:
            if predicted[index] == target.long()[index]:
                correct5 += 1

        loss_mean += loss.item()
        train_loss_curve.append(loss.item())
        train_acc_curve.append(correct / total)

        if (i + 1) % 100 == 0:
            loss_mean = loss_mean / 100
            print(
                "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3} Loss:{:.4f} Acc:{:.2%}".format(epoch, EPOCH,
                                                                                                      i + 1, len(
                        train_loader), loss_mean, correct / total))
            loss_mean = 0
    if total0 == 0:
        train_acc_curve0.append(0.0)
    else:
        train_acc_curve0.append(correct0 / total0)

    if total1 == 0:
        train_acc_curve1.append(0.0)
    else:
        train_acc_curve1.append(correct1 / total1)

    if total2 == 0:
        train_acc_curve2.append(0.0)
    else:
        train_acc_curve2.append(correct2 / total2)

    if total3 == 0:
        train_acc_curve3.append(0.0)
    else:
        train_acc_curve3.append(correct3 / total3)

    if total4 == 0:
        train_acc_curve4.append(0.0)
    else:
        train_acc_curve4.append(correct4 / total4)

    if total5 == 0:
        train_acc_curve5.append(0.0)
    else:
        train_acc_curve5.append(correct5 / total5)

    scheduler.step()  # 更新学习率

    # 验证模型
    if (epoch + 1) % 1 == 0:
        correct_val = 0
        total_val = 0

        correct0_val = 0
        correct1_val = 0
        correct2_val = 0
        correct3_val = 0
        correct4_val = 0
        correct5_val = 0

        total0_val = 0
        total1_val = 0
        total2_val = 0
        total3_val = 0
        total4_val = 0
        total5_val = 0

        loss_val = 0
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                deep_data, target = data[0][:, 34:-1], data[0][:, -1]
                deep_data = deep_data.to(device)
                target = target.to(device)
                deep_data = deep_data.view(128, 1, 39, 100)
                outputs = model(deep_data)
                loss = criterion(outputs, target.long())

                _, predicted = torch.max(outputs.data, 1)
                for tar_val in target.long().cpu().numpy():
                    valid_real_label.append(tar_val)
                for predict_val in predicted.cpu().numpy():
                    valid_pre_label.append(predict_val)

                total_val += target.size(0)
                correct_val += (predicted == target.long()).cpu().squeeze().sum().numpy()

                loss_val += loss.item()

                type_val_index0 = np.where(target.cpu().numpy() == 0.0)
                total0_val += len(type_val_index0[0])
                for index in type_val_index0[0]:
                    if predicted[index] == target.long()[index]:
                        correct0_val += 1

                type_val_index1 = np.where(target.cpu().numpy() == 1.0)
                total1_val += len(type_val_index1[0])
                for index in type_val_index1[0]:
                    if predicted[index] == target.long()[index]:
                        correct1_val += 1

                type_val_index2 = np.where(target.cpu().numpy() == 2.0)
                total2_val += len(type_val_index2[0])
                for index in type_val_index2[0]:
                    if predicted[index] == target.long()[index]:
                        correct2_val += 1

                type_val_index3 = np.where(target.cpu().numpy() == 3.0)
                total3_val += len(type_val_index3[0])
                for index in type_val_index3[0]:
                    if predicted[index] == target.long()[index]:
                        correct3_val += 1

                type_val_index4 = np.where(target.cpu().numpy() == 4.0)
                total4_val += len(type_val_index4[0])
                for index in type_val_index4[0]:
                    if predicted[index] == target.long()[index]:
                        correct4_val += 1

                type_val_index5 = np.where(target.cpu().numpy() == 5.0)
                total5_val += len(type_val_index5[0])
                for index in type_val_index5[0]:
                    if predicted[index] == target.long()[index]:
                        correct5_val += 1

            valid_loss_curve.append(loss_val / valid_loader.__len__())
            valid_acc_curve.append(correct_val / total_val)
            print(
                "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3} Loss:{:.4f} Acc:{:.2%}".format(epoch, EPOCH,
                                                                                                      j + 1, len(
                        valid_loader), loss_val / valid_loader.__len__(), correct_val / total_val))

            if total0_val == 0:
                valid_acc_curve0.append(0.0)
            else:
                valid_acc_curve0.append(correct0_val / total0_val)

            if total1_val == 0:
                valid_acc_curve1.append(0.0)
            else:
                valid_acc_curve1.append(correct1_val / total1_val)

            if total2_val == 0:
                valid_acc_curve2.append(0.0)
            else:
                valid_acc_curve2.append(correct2_val / total2_val)

            if total3_val == 0:
                valid_acc_curve3.append(0.0)
            else:
                valid_acc_curve3.append(correct3_val / total3_val)

            if total4_val == 0:
                valid_acc_curve4.append(0.0)
            else:
                valid_acc_curve4.append(correct4_val / total4_val)

            if total5_val == 0:
                valid_acc_curve5.append(0.0)
            else:
                valid_acc_curve5.append(correct5_val / total5_val)

with open("./results/train_neg0.txt", mode='w', encoding='utf-8') as f0:
    for val in train_acc_curve0:
        f0.write(str(val))
        f0.write('\n')

with open("./results/train_neg1.txt", mode='w', encoding='utf-8') as f1:
    for val in train_acc_curve1:
        f1.write(str(val))
        f1.write('\n')

with open("./results/train_neg2.txt", mode='w', encoding='utf-8') as f2:
    for val in train_acc_curve2:
        f2.write(str(val))
        f2.write('\n')

with open("./results/train_neg3.txt", mode='w', encoding='utf-8') as f3:
    for val in train_acc_curve3:
        f3.write(str(val))
        f3.write('\n')

with open("./results/train_neg4.txt", mode='w', encoding='utf-8') as f4:
    for val in train_acc_curve4:
        f4.write(str(val))
        f4.write('\n')

with open("./results/train_pos.txt", mode='w', encoding='utf-8') as f5:
    for val in train_acc_curve5:
        f5.write(str(val))
        f5.write('\n')

with open("./results/valid_neg0.txt", mode='w', encoding='utf-8') as f6:
    for val in valid_acc_curve0:
        f6.write(str(val))
        f6.write('\n')

with open("./results/valid_neg1.txt", mode='w', encoding='utf-8') as f7:
    for val in valid_acc_curve1:
        f7.write(str(val))
        f7.write('\n')

with open("./results/valid_neg2.txt", mode='w', encoding='utf-8') as f8:
    for val in valid_acc_curve2:
        f8.write(str(val))
        f8.write('\n')

with open("./results/valid_neg3.txt", mode='w', encoding='utf-8') as f9:
    for val in valid_acc_curve3:
        f9.write(str(val))
        f9.write('\n')

with open("./results/valid_neg4.txt", mode='w', encoding='utf-8') as f10:
    for val in valid_acc_curve4:
        f10.write(str(val))
        f10.write('\n')

with open("./results/valid_pos.txt", mode='w', encoding='utf-8') as f11:
    for val in valid_acc_curve5:
        f11.write(str(val))
        f11.write('\n')

with open("./results/train_loss.txt", mode='w', encoding='utf-8') as f12:
    for val in train_loss_curve:
        f12.write(str(val))
        f12.write('\n')

with open("./results/valid_loss.txt", mode='w', encoding='utf-8') as f13:
    for val in valid_loss_curve:
        f13.write(str(val))
        f13.write('\n')

with open("./results/train_target.txt", mode='w', encoding='utf-8') as f14:
    for val in train_real_label:
        f14.write(str(val))
        f14.write('\n')

with open("./results/train_pre.txt", mode='w', encoding='utf-8') as f15:
    for val in train_pre_label:
        f15.write(str(val))
        f15.write('\n')

with open("./results/valid_target.txt", mode='w', encoding='utf-8') as f16:
    for val in valid_real_label:
        f16.write(str(val))
        f16.write('\n')

with open("./results/valid_pre.txt", mode='w', encoding='utf-8') as f17:
    for val in valid_pre_label:
        f17.write(str(val))
        f17.write('\n')
with open("./results/train_acc.txt", mode='w', encoding='utf-8') as f18:
    for val in train_acc_curve:
        f18.write(str(val))
        f18.write('\n')
with open("./results/valid_acc.txt", mode='w', encoding='utf-8') as f19:
    for val in valid_acc_curve:
        f19.write(str(val))
        f19.write('\n')
