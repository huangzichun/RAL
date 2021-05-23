# 管理有标签数据和无标签数据

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

class Data():
    def __init__(self, data_input, embed):
        self.n_word = data_input.n_word
        self.data_input = data_input
        self.embed = embed

        # 有标签数据和无标签数据
        self.labeled_data_list =  np.empty((self.n_word, 2))
        self.labeled_target_list = np.empty(self.n_word) 
        self.labeled_record = np.empty(self.n_word)
        self.labeled_num = 0
        self.unlabeled_data_set = set(range(self.n_word))

    def train_loader(self, data_input): # 返回所有带标签数据的embed
        data_list = []
        target_list = np.empty(self.labeled_num)
        
        for i in range(self.labeled_num):
            target_list[i] = self.labeled_target_list[i]
            data_add = []
            for j in range(2):
                data_add = data_add + self.embed.id2embed(self.labeled_data_list[i][j])
            data_list.append(data_add)
        
        data_list = np.array(data_list)
        data_list = Variable(torch.from_numpy(data_list)).type(torch.FloatTensor)
        x = data_list
        y = Variable(torch.from_numpy(target_list)).type(torch.LongTensor)
        train_data = TensorDataset(x, y)
        loader = DataLoader(dataset = train_data, batch_size = 16, shuffle = True)

        return loader 
        
    def choose_and_update(self, action): # 根据agent action返回值去选择对应数据,并更新数据
        input_data, input_target = self.data_input.data_list[action], self.data_input.target_list[action]
        
        self.labeled_record[action] = 1
        self.labeled_data_list[self.labeled_num] = input_data
        self.labeled_target_list[self.labeled_num] = input_target
        self.labeled_num += 1
        # print(self.unlabeled_data_set)
        self.unlabeled_data_set.remove(action)
        
        return input_data, input_target


    def value_filter(self, actions_value): # 去除已选过的action
        actions_value_new = actions_value
        min_value = min(actions_value)
        for i in range(self.n_word):
            if self.labeled_record[i] > 0.5:
                actions_value_new[i] = min_value - 1

        return actions_value_new
    
    def reset(self, ): # 重置有标签数据和无标签数据
        self.unlabeled_data_set = set(range(self.n_word))
        self.labeled_data_list =  np.empty((self.n_word, 2)) 
        self.labeled_target_list = np.empty(self.n_word)
        self.labeled_record = np.empty(self.n_word)
        self.labeled_num = 0

