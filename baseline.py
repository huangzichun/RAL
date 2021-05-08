import json
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import data_processing

class Net(nn.Module):
    def __init__(self, n_dict, EMBEDDING_DIM):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(n_dict, EMBEDDING_DIM)
        self.out = nn.Linear(2 * EMBEDDING_DIM, 1)

    def forward(self, x):
        emb = self.embedding(x)
        # print(x.size())
        emb = emb.view(-1, 2 * EMBEDDING_DIM)
        # print(emb.size())
        out = self.out(emb)
        out = F.sigmoid(out)
        return out

FILENAME = 'coordpairs_wiki100.json'
TESTFILENAME = 'coordpairs_wiki100.json'
EMBEDDING_DIM = 256


data_input = data_processing.data_processing(FILENAME, TESTFILENAME, EMBEDDING_DIM)
n_dict = data_input.n_dict
n_word = data_input.n_word
print(n_dict)
# n_word = 1

data_list = Variable(torch.from_numpy(data_input.data_list)).type(torch.LongTensor)
x = data_list
y = Variable(torch.from_numpy(data_input.target_list)).type(torch.FloatTensor)
train_data = TensorDataset(x, y)
loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True)

net = Net(n_dict, EMBEDDING_DIM)
loss_func = nn.MSELoss()
epoch = 1
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

for i in range(epoch):
    error = 0
    for batch_id, batch_data in enumerate(loader):
        batch_x, batch_y = batch_data
        optimizer.zero_grad()
        batch_out = net.forward(batch_x)
        loss = loss_func(batch_out, batch_y)
        loss.backward()
        optimizer.step()

    for j in range(n_word):
        x_input = x[j]
        target = y[j]
        out = net.forward(x_input)
        if abs(out - target) > 0.5:
            error += 1
    # print('epoch:', i, 'loss:', float(loss), 'acc:', 1 - error/n_word)
