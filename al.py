# 规定al的策略 目前采用一些al的统计量

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class al(object):
    def __init__(self, data, data_input, embed, Model):
        self.Model = Model
        self.data = data
        self.embed = embed
        self.data_list = data_input.data_list
        self.unlabeled_data_set = data.unlabeled_data_set

        self.n_al = 1 # al统计量个数
        self.al_data_list = np.empty((data.n_word, self.n_al))

    def al_uncertain(self, ):  # 模型输出，衡量不确信度
        length = len(self.data_list)
        new_data_list = []
        for i in range(length):
            # print(self.embed.id2embed(int(self.data_list[i][0])))
            new_data_list.append(self.embed.id2embed(int(self.data_list[i][0])) + self.embed.id2embed(int(self.data_list[i][1])))
        
        new_data_list = np.array(new_data_list)
        new_data_list = Variable(torch.from_numpy(new_data_list)).type(torch.FloatTensor)
        al_uncertain = self.Model.net.forward(new_data_list).detach().numpy()
        # print(al_uncertain)
        return(al_uncertain)

    # def al_gradient_change(self, ):  # 梯度变化

    # def al_loss_net(self, ): # loss网络

    def update(self, ):  # 更新AL统计量
        # 更新不确信度
        al_uncertain = self.al_uncertain()
        for i in range(self.data.n_word):
            if i in self.unlabeled_data_set:
                self.al_data_list[i] = abs(al_uncertain[i][0] - al_uncertain[i][1]) #这里采用的是margin
            else:
                self.al_data_list[i] = 0

        return self.al_data_list
