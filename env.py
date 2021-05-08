# 基于上下位词标注的环境

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class env(object):
    def __init__(self, data, data_input, al, embed, Model):
        self.action_space_dim = data_input.n_word
        self.state = self.initial(al)  # 初始状态，为一个n_word * al.num的numpy矩阵
        self.long_reward_counter = 0
        self.gamma0 = 100
        self.data = data
        self.al = al
        self.Model = Model
        self.embed = embed

    def feedback(self, action):  # 对agent采取的action进行反馈
        input_data, input_target = self.data.choose_and_update(action) # action为选中数据的下标，并更新数据集
        x1_embed = self.embed.id2embed(int(input_data[0]))
        x2_embed = self.embed.id2embed(int(input_data[1]))
        x_input = np.array((x1_embed + x2_embed))
        x = Variable(torch.from_numpy(x_input)).type(torch.FloatTensor)
        predict_label = self.Model.net.forward(x).detach().numpy().squeeze(0)
        
        # print(predict_label)
        short_reward = long_reward = 0
        if abs(input_target - predict_label) < 0.5: # 短期reward，希望能挑选出那些机器会判断错的数据
            short_reward = 1
        
        if self.long_reward_counter % 5 == 0:  # 定期回传long reward
            long_reward = self.Model.acc_change()
        
        self.long_reward_counter += 1
        r = short_reward + self.gamma0 * long_reward
        # print(short_reward, long_reward, r)

        # 更新模型
        self.Model.train()

        # 状态改变（重新计算al统计量）
        self.state = self.al.update()
        s_next = self.state 
        return s_next, r
    
    def initial(self, al):
        return al.update()




