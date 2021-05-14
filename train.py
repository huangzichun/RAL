# 模型训练部分

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def train(model, data, agent, al, epoch, env, budget, MEMORY_CAPACITY):
    for i_episode in range(epoch):
        print('epoch:', i_episode + 1)
        data.reset()
        s = env.initial(al) # 得到初始状态
        for j in range(budget):
            # 根据dqn来接受现在的状态，得到一个行为
            s = Variable(torch.from_numpy(s)).type(torch.FloatTensor)
            a = agent.choose_action(s) 
            s_next, r = env.feedback(a) # 根据环境的行为，给出一个反馈
            
            agent.store_transition(s,a,r,s_next) # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
            
            if agent.memory_counter > MEMORY_CAPACITY:
                agent.Learn()

            s = s_next 
        
        torch.save(agent.target_net, '\model.pkl')