# 规定al的策略 目前采用一些al的统计量

import numpy as np 
import torch
from torch.autograd import Variable

class al(object):
    def __init__(self, data, data_input, Model):
        self.Model = Model
        self.data = data
        self.data_list = Variable(torch.from_numpy(data_input.data_list)).type(torch.LongTensor)
        self.unlabeled_data_set = data.unlabeled_data_set
        self.n_al = 1
        self.al_data_list = np.empty((data.n_word, self.n_al))

    def al_uncertain(self, ):  # 不确信度
        al_uncertain = self.Model.net.forward(self.data_list).detach().numpy()
        print(al_uncertain)
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

