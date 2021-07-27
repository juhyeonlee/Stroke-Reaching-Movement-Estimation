import torch
import torch.nn as nn
import numpy as np


torch.manual_seed(0)
np.random.seed(0)

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(50, 100)
        # self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, output_dim)
        self.out_act = nn.Softmax()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        # a2 = self.fc2(dout)
        # h2 = self.prelu(a2)
        y = self.out(dout)
        # y = self.out_act(a3)
        return y


def train_NN(model, X, y, lr=1e-3, epoch=100):
    batch_size = 32
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for t in range(epoch):
        loss_epoch = []
        rand_order = np.random.permutation(len(X))
        X = X[rand_order, :]
        y = y[rand_order]
        # if epoch == 50:
        #     lr = lr/10
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for beg_i in range(0, X.shape[0], batch_size):
            X_batch = X[beg_i:beg_i + batch_size, :]
            y_batch = y[beg_i:beg_i + batch_size]
            X_batch = torch.from_numpy(X_batch).type(torch.FloatTensor)
            y_batch = torch.from_numpy(y_batch)

            model.zero_grad()
            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            loss_epoch.append(loss.item())

            loss.backward()
            optimizer.step()
        # print(np.mean(loss_epoch))

def test_NN(model, X):
    model.eval()
    y_pred_score = model(torch.from_numpy(X).type(torch.FloatTensor))
    _, y_pred = torch.max(y_pred_score, 1)
    return y_pred.detach().numpy(), y_pred_score.detach().numpy()


class autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.en1 = nn.Linear(input_dim, 100)
        self.en2 = nn.Linear(100, 10)
        self.de2 = nn.Linear(10, 100)
        self.de3 = nn.Linear(100, input_dim)
        self.relu = nn.ReLU()
        self.dout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10, 2)

    def forward(self, input_):
        en_out = self.en1(input_)
        feature = self.en2(en_out)
        de_out = self.de2(feature)
        de_out = self.de3(de_out)
        feature_d = self.dout(feature)
        classify_out = self.fc1(feature_d)
        return de_out, classify_out
