# 机器分类器

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# 分类器网络，输入word embedding（不训练），输出分类结果，一层的网络
class Classify_Net(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super(Classify_Net, self).__init__()
        # self.EMBEDDING_DIM = EMBEDDING_DIM
        # self.embedding = nn.Embedding(n_dict, EMBEDDING_DIM)
        self.fc1 = nn.Linear(2 * EMBEDDING_DIM, 20)
        self.out = nn.Linear(20, 2)

    def forward(self, x):
        # emb = self.embedding(x)
        # emb = emb.view(-1, 2 * self.EMBEDDING_DIM)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.out(x)
        return output

class model(object):
    def __init__(self, data, data_processer, Embedding, EMBEDDING_DIM):
        self.data_processer = data_processer
        self.net = Classify_Net(EMBEDDING_DIM)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = 0.01)
        self.loss_func = nn.CrossEntropyLoss() 
        self.epoch = 200
        self.embed = Embedding
        self.data = data
        self.acc = self.test()
        

    def train(self, ): # 使用所有有标签数据训练
        loader = self.data.train_loader(self.data_processer)
        for i in range(self.epoch):
            total_loss = 0 
            for batch_id, batch_data in enumerate(loader):
                batch_x, batch_y = batch_data
                self.optimizer.zero_grad()
                batch_out = self.net.forward(batch_x)
                loss = self.loss_func(batch_out, batch_y)
                total_loss += float(loss)
                loss.backward()
                self.optimizer.step()

            # if i % 50 == 0:
            #     print(i, "epoch  loss:", total_loss / self.data.labeled_num)
    
    def test(self, ): # 测试模型准确率上升
        acc_num = 0
        num = 0
        test_data_list = self.data_processer.test_data_list
        test_target_list = self.data_processer.test_target_list

        for i in range(len(test_data_list)):
            num += 1
            x = test_data_list[i]
            x1_embed = self.embed.id2embed(int(x[0]))
            x2_embed = self.embed.id2embed(int(x[1]))
            x_input = np.array((x1_embed + x2_embed))
            x_input = Variable(torch.from_numpy(x_input)).type(torch.FloatTensor)
            
            y = test_target_list[i]
            y_predict = self.net.forward(x_input)
            
            if (y_predict[0] - y_predict[1]) * (y - 0.5) < 0:
                acc_num += 1
        
        return acc_num/num
    
    def acc_change(self, ):
        new_acc = self.test()
        acc_change = new_acc - self.acc
        self.acc = new_acc
        return acc_change


