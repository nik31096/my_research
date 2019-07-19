import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np
import pickle
import sys
from matplotlib import pyplot as plt


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def preprocessing(data_path):
    batch_list = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5",]
    trainX, trainY = [], []
    for i in range(len(batch_list)):
        train = unpickle(data_path + '/' + "{}".format(batch_list[i]))
        trainX.append(np.array(train.get(b'data', [])).reshape(10000, 3, 32, 32))
        trainY.append(np.array(train.get(b'labels', [])))
    trainX = (np.array(trainX).astype("float") / 255.0).reshape((50000, 3, 32, 32))
    trainY = np.array(trainY).reshape((50000,))
    test = unpickle(data_path + '/' + "test_batch")
    testX = (np.array(test.get(b"data")).astype("float") / 255.0).reshape((10000, 3, 32, 32))
    testY = np.array(test.get(b"labels", []))

    return trainX, trainY, testX, testY

    
# building network
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        conv1filters = 16
        conv2filters = 32
        conv3filters = 64
        conv4filters = 128
        conv5filters = 256
        self.conv1 = nn.Conv2d(in_channels=trainX.shape[1], out_channels=conv1filters, kernel_size=(5, 5), stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=conv1filters)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(conv1filters, conv2filters, (5, 5), stride=1)
        self.bn2 = nn.BatchNorm2d(conv2filters)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.conv3 = nn.Conv2d(conv2filters, conv3filters, (5, 5), stride=1)
        self.bn3 = nn.BatchNorm2d(conv3filters)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.conv4 = nn.Conv2d(conv3filters, conv4filters, (5, 5), stride=1)
        self.bn4 = nn.BatchNorm2d(conv4filters)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=1)
        self.conv5 = nn.Conv2d(conv4filters, conv5filters, (5, 5), stride=1)
        self.dense1 = nn.Linear(4*4*conv5filters, 500)
        self.dropout = nn.Dropout()
        self.dense2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)  # (32 - 5 + 2*0) / 1 + 1 = 28
        x = self.bn1(x)
        x = self.pool1(x)  # (30 - 3 + 2*0) / 1 + 1 = 26
        x = F.relu(x)  # the same
        x = self.conv2(x)  # (26 - 5 + 2*0) / 1 + 1 = 22
        x = self.bn2(x)
        x = self.pool2(x)  # (22 - 3 + 2*0) / 1 + 1 = 20
        x = F.relu(x)  # the same
        x = self.conv3(x)  # (20 - 5 + 2*0) / 1 + 1 = 16
        x = self.bn3(x)
        x = self.pool3(x)  # (16 - 3 + 2*0) / 1 + 1 = 14
        x = F.relu(x)  # the same
        x = self.conv4(x)  # (14 - 5 + 2*0) / 1 + 1 = 10
        x = self.bn4(x)
        x = self.pool4(x)  # (10 - 3 + 2*0) / 1 + 1 = 8
        x = F.relu(x)  # the same
        x = self.conv5(x)  # (8 - 5 + 2*0) / 1 + 1 = 4 
        x = F.relu(x)  # the same
        x = x.flatten(1, )  # flatten in vector (batch_size, 4*4*conv5filters)
        x = self.dense1(x)  # batch_size * 4*4*256
        x = self.dropout(F.relu(x))
        output = self.dense2(x)  # batch_size*10

        return output

# TODO train model and save checkpoint to clear GPU memory
# And then load model checkpoint in another file to get train and test accuracy

if __name__ == '__main__':
    print("[INFO] loading dataset")
    # importing data
    path_to_cifar_dataset = './cifar10/cifar-10-batches-py'  # '/home/nik-96/Documents/cifar10/cifar-10-batches-py'
    trainX, trainY, testX, testY = preprocessing(path_to_cifar_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    print(f"Device: {device}")
    model = network().to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    batch_size = 128
    train_dataset = TensorDataset(torch.tensor(trainX, dtype=torch.float32), torch.tensor(trainY, dtype=torch.int64))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(testX, dtype=torch.float32), torch.tensor(testY, dtype=torch.int64))
    test_loader = DataLoader(test_dataset, batch_size=500)
    epochs = 20
    model.train(True)
    print("[INFO] training")
    num_elements = len(train_dataset)
    for epoch in range(epochs):
        for x, y in train_loader:
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            opt.step()
            opt.zero_grad()
        if epoch % 2 == 0:
            # loss is a scalar so we can use loss.item() to see loss value
            print("epoch #{}, loss: {}".format(epoch, loss.item()))

    indices = np.random.randint(0, trainX.shape[0], size=3000)
    y_pred = model(torch.tensor(trainX[indices], dtype=torch.float32).to(device)) 
    print("train accuracy:", np.mean(torch.argmax(y_pred, 1).cpu().data.numpy() == trainY[indices]))
    print("[INFO] saving model")
    torch.save(model.state_dict(), './model.pt')
    print("[INFO] testing")
    accuracy = []
    model.train(False)
    with torch.no_grad():  # for not having extra info like gradients
        for x, y in test_loader:
            test_y_pred = model(x.to(device))
            accuracy.append(np.sum(torch.argmax(test_y_pred, 1).cpu().data.numpy() == y.data.numpy()))

    print("test accuracy:", np.sum(accuracy) / testY.shape[0])

