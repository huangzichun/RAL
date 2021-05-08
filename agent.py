# 模型中的agent部分
# 需要改的地方：状态和动作空间的维数在变化，怎么处理需要订正

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# 定义一个判断数据是否可学的网络类（暂未使用）
class Net(nn.Module):
    def __init__(self, ):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(LEARNABLE_FEATURE_DIM, dim_2)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(dim_2, 2)
        self.out.weight.data.normal_(0, 0.1)  
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        learnable = self.out(x)
        return learnable


# 定义一个Q网络的类，输入：当前状态；输出：每种action能获得的return
# 对于那些已经采取过的行动
class Q_Net(nn.Module):
    def __init__(self, N_STATES, N_ACTIONS):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 64)
        self.fc1.weight.data.normal_(0, 0.1)  
        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, Data, N_STATES, N_ACTIONS):
        # DQN有两个神经网络，一个是eval_net一个是target_net
        # 两个神经网络相同，参数不同，每隔一段时间把eval_net的参数转化成target_net的参数，产生延迟的效果
        self.eval_net,self.target_net = Q_Net(N_STATES, N_ACTIONS), Q_Net(N_STATES, N_ACTIONS)

        self.TARGET_REPLACE_ITER = 100
        self.BATCH_SIZE = 32
        self.LR = 0.01
        self.MEMORY_CAPACITY = 2000
        self.GAMMA = 0.9

        self.learn_step_counter = 0 # 学习步数计数器
        self.memory_counter = 0 # 记忆库中位值的计数器
        self.memory = np.zeros((self.MEMORY_CAPACITY,N_STATES * 2 + 2)) # 初始化记忆库
        # 记忆库初始化为全0，存储两个state的数值加上一个a(action)和一个r(reward)的数值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr = self.LR)
        self.loss_func = nn.MSELoss() 
        self.epsilon = 0.95
        self.N_STATES = N_STATES
        self.Data = Data
    
    # 接收环境中的观测值，并采取动作
    def choose_action(self, x):
        # print(type(x))
        x = x.squeeze(-1)
        # print(type(x))
        # x = x.view(-1, self.N_STATES)
        if np.random.uniform() < self.epsilon: # epsilon-greedy策略去选择动作
            # print(x)
            actions_value = self.eval_net.forward(x).detach().numpy() # 返回的是action的值
            # print(actions_value)
            # actions_value = actions_value.squeeze(-1)
            actions_value = self.Data.value_filter(actions_value) # 过滤掉已经不能再次选择的动作（直接令为0）
            action = np.argmax(actions_value)
            # print("max!")
        else:
            action = random.sample(self.Data.unlabeled_data_set, 1)
            action = action[0]
            # print("random!")
        print(action)
        return action    

    
    #记忆库，存储之前的记忆，学习之前的记忆库里的东西
    def store_transition(self, s, a, r, s_next):
        s = s.squeeze(-1)
        s_next = s_next.squeeze(-1)
        transition = np.hstack((s, [a, r], s_next))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def Learn(self):
        # target net 参数更新,每隔 TARGET_REPLACE_ITER 更新一下
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :] 
        
        # 打包记忆，分开保存进b_s，b_a，b_r，b_s
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]) #当前state
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)) #action
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]) # reward
        b_s_next = torch.FloatTensor(b_memory[:, -self.N_STATES:]) #下一个state

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_next).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward() #误差反向传播
        self.optimizer.step()
    