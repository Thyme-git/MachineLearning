import torch
from torch import nn

class Classifer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, (3, 3), 1)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(10*10*64, 256)
        self.relu3 = nn.ReLU()

        self.out = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, X):
        X = X.view(X.shape[0], 1, 28, 28)
        X = self.maxpool(self.batchnorm1(self.relu1(self.conv1(X))))
        X = self.relu2(self.conv2(X))

        X = X.view(X.shape[0], -1)
        X = self.relu3(self.fc(X))

        return self.softmax(self.out(X))

if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # device = (
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    # else "cpu"
    # )

    device = 'cpu'

    train_data = torch.from_numpy(np.array(pd.read_csv('./train.csv'))).to(torch.float32)
    test_data = torch.from_numpy(np.array(pd.read_csv('./test.csv'))).to(torch.float32)

    y_train, X_train = train_data[:, 0].long(), train_data[:, 1:]
    y_test, X_test = test_data[:, 0].long(), test_data[:, 1:]

    X_train /= 255
    X_test /= 255

    # print(X_train.shape, y_train.shape)

    clf = Classifer().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    clf.train()
    for _ in range(100):
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        y_pred = clf(X_train)
        loss = loss_fn(y_pred, y_train)

        acc = (y_pred.argmax(1) == y_train).sum() / len(y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
        print('acc:', acc.item())