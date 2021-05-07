# 机器分类器

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# 分类器网络
class Classify_Net(nn.Module):
    def __init__(self, n_dict, EMBEDDING_DIM):
        super(Classify_Net, self).__init__()
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.embedding = nn.Embedding(n_dict, EMBEDDING_DIM)
        self.out = nn.Linear(2 * EMBEDDING_DIM, 1)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(-1, 2 * self.EMBEDDING_DIM)
        out = self.out(emb)
        out = F.sigmoid(out)
        return out

class model():
    def __init__(self, data, data_processer, EMBEDDING_DIM):
        self.data_processer = data_processer
        self.net = Classify_Net(data_processer.n_dict, EMBEDDING_DIM)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.01)
        self.loss_func = nn.MSELoss() 
        self.epoch = 200  
        self.acc = self.test()
        self.data = data
    
    def train(self, ): # 使用所有有标签数据训练
        loader = self.data.train_loader(self.data_processer)
        for i in range(self.epoch):
            for batch_id, batch_data in enumerate(loader):
                batch_x, batch_y = batch_data
                self.optimizer.zero_grad()
                batch_out = self.net.forward(batch_x)
                loss = self.loss_func(batch_out, batch_y)
                loss.backward()
                self.optimizer.step()
    
    def test(self, ): # 测试模型准确率上升
        acc_num = 0
        num = 0
        test_data_list = self.data_processer.test_data_list
        test_target_list = self.data_processer.test_target_list
        test_data_list = Variable(torch.from_numpy(test_data_list)).type(torch.LongTensor)
        test_target_list = Variable(torch.from_numpy(test_target_list)).type(torch.FloatTensor)

        for i in range(len(test_data_list)):
            num += 1
            x = test_data_list[i]
            y = test_target_list[i]
            y_predict = self.net.forward(x)
            
            if abs(y_predict - y) < 0.5:
                acc_num += 1
        
        return acc_num/num
    
    def acc_change(self, ):
        new_acc = self.test()
        acc_change = new_acc - self.acc
        self.acc = new_acc
        return acc_change


