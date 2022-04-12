import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from xgboost import train

np.random.seed(13)

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01


class SimpleNewsNet(nn.Module):
    def __init__(self, input_dim, drop=0.1, hidden=32):
        super(SimpleNewsNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=drop)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc



def main():
    path = 'new data\\new_df_train.csv'
    df = pd.read_csv(path)
    X = df.drop('is_fake', axis=1)
    y = df['is_fake']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=13)
    train_data = TrainData(torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(y_train.to_numpy()))
    test_data = TestData(torch.FloatTensor(X_test.to_numpy()))

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    model = SimpleNewsNet(X_train.shape[1], hidden=128, drop=0.25)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    print(classification_report(y_test, y_pred_list))


main()
