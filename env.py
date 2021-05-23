# 基于上下位词标注的环境

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class env(object):
    def __init__(self, data, data_input, al, embed, Model):
        self.action_space_dim = data_input.n_word
        self.state = self.initial(al)  # 初始状态，为一个 n_word * al.num的numpy矩阵
        self.counter = 0
        self.gamma0 = 100 # 长期reward的权重

        self.data = data
        self.al = al
        self.Model = Model
        self.embed = embed

    def feedback(self, action, queried_by_who_action):  # 对agent采取的action进行反馈
        input_data, input_target = self.data.choose_and_update(action, queried_by_who_action) # action为选中数据的下标，并更新数据集
        short_reward = long_reward = 0

        if queried_by_who_action == 1:
            # queried by human shall suffer
            short_reward = -1e-1

            x1_embed = self.embed.id2embed(int(input_data[0]))
            x2_embed = self.embed.id2embed(int(input_data[1]))
            x_input = np.array((x1_embed + x2_embed))
            x = Variable(torch.from_numpy(x_input)).type(torch.FloatTensor)
            predict_label = self.Model.net.forward(x).detach().numpy()
            input_target_pred = np.argmax(predict_label)

            if input_target_pred != input_target:
                short_reward += abs(predict_label[0] - predict_label[1])
            else:
                short_reward += -1e-1


        # if (predict_label[0] - predict_label[1]) * (input_target - 0.5) > 0: # 短期reward，希望能挑选出那些机器会判断错的数据
        #     short_reward += 1

        if self.counter % 32 == 0:  # 定期回传long reward
            self.Model.train()
            long_reward += self.Model.acc_change()
            acc = self.Model.test()
            print('acc = {} and #labeled_sample= {}'.format(acc, self.data.labeled_num_by_human))

        self.counter += 1
        r = short_reward + self.gamma0 * long_reward

        # 状态改变（重新计算al统计量）
        self.state = self.al.update()
        s_next = self.state
        return s_next, r
    
    def initial(self, al):
        return al.update()

    def get_rest_unlabeled_data_effect(self):
        acc_num = 0
        unlabeled_data_index = self.data.get_unlabeled_data()
        num = len(unlabeled_data_index)
        for ind in unlabeled_data_index:
            x, input_target = self.data.get_feat_label_by_index(ind)
            predict_label = self.Model.net.forward(x).detach().numpy()

            if (predict_label[0] - predict_label[1]) * (input_target - 0.5) < 0:
                acc_num += 1

        return acc_num/num, num

    def get_test_unlabeled_data_effect(self):
        acc_num = 0
        num = 0
        test_data_list = self.data.data_input.test_data_list
        test_target_list = self.data.data_input.test_target_list

        for i in range(len(test_data_list)):
            num += 1
            x = test_data_list[i]
            x1_embed = self.embed.id2embed(int(x[0]))
            x2_embed = self.embed.id2embed(int(x[1]))
            x_input = np.array((x1_embed + x2_embed))
            x_input = Variable(torch.from_numpy(x_input)).type(torch.FloatTensor)

            y = test_target_list[i]
            y_predict = self.Model.net.forward(x_input)

            if (y_predict[0] - y_predict[1]) * (y - 0.5) < 0:
                acc_num += 1

        return acc_num/num, num


