# 规定al的策略 目前采用一些al的统计量

import numpy as np 
import torch
from torch.autograd import Variable

class al(object):
    def __init__(self, data, data_input, embed, Model):
        self.Model = Model
        self.data = data
        self.embed = embed
        self.data_list = data_input.data_list
        self.unlabeled_data_set = data.unlabeled_data_set
        self.n_al = 1
        self.al_data_list = np.empty((data.n_word, self.n_al))

    def al_uncertain(self, ):  # 不确信度
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
    
    def update(self, ):
        al_uncertain = self.al_uncertain()
        for i in range(self.data.n_word):
            if i in self.unlabeled_data_set:
                self.al_data_list[i] = 0
            else:
                self.al_data_list[i] = al_uncertain[i]
        return self.al_data_list


    # def al_gradient_change(self, ):  # 梯度变化

    # def al_loss_net(self, ): # loss网络

