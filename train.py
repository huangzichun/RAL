# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def train(model, agent, al, epoch, env, budget, MEMORY_CAPACITY):
    # learnable_net = agent.Net() 判断可学性的网络，可以后续添加
    for i_episode in range(epoch):
        s = env.initial(al) # 得到初始状态
        for j in range(budget):
            # 根据dqn来接受现在的状态，得到一个行为
            # 行为：挑选数据 
            s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
            a = agent.choose_action(s) 
            s_next, r = env.feedback(a) # 根据环境的行为，给出一个反馈
            # if j > 10: # 如果积累了一些数据，就开始训练
            
            agent.store_transition(s,a,r,s_next) # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
            
            if agent.memory_counter > MEMORY_CAPACITY:
               # print("I'm learning!")
                agent.Learn()
            
            acc = model.test()
            print(acc, r)
            
            s = s_next # 现在的状态赋值到下一个状态上去