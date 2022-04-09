import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np

class MLP(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            state_dim: 输入的特征数即环境的状态维度
            action_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim) # 输出层
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQN:
    
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.frame_idx = 0
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(-1.0 * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim = cfg.hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim = cfg.hidden_dim).to(self.device)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
            
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)
    
    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_values = self.policy_net(state)     # 计算Q(s, a)状态-动作价值函数。
                # 注意与策略梯度的神经网络输出区别：DQN输出的是每个动作的价值，而策略梯度输出的是每个动作的概率分布
                action = q_values.max(1)[1].item()   # max(1)返回时一个元组，第一个元素时值，第二个元素时值对应的索引
                # print('q_values: ', q_values)
        else:
            action = random.randrange(self.action_dim)
        return action
    
    def update(self):
        if len(self.memory) < self.batch_size: # 当memory中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # 转为张量
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values_temp = self.policy_net(state_batch)   # 神经网络输出的是每个动作对应的价值
        q_values = q_values_temp.gather(dim=1, index=action_batch) # 计算当前状态(s_t,a)对应的Q(s_t, a)，即选择实际发生动作对应的价值
        next_q_values_temp = self.target_net(next_state_batch)
        next_q_values = next_q_values_temp.max(1)[0].detach() # 计算下一时刻的状态(s_t_,a)对应的Q值
        
        # print('q_values_temp:', q_values_temp)
        # print('q_values_:', q_values)
        # print('next_q_values_temp:', next_q_values_temp)
        # print('next_q_values:', next_q_values)
        
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)   #torch.clamp(input, min, max, out=False) -> Tensor, 将输入input张量每个元素的夹紧到区间 [min,max]
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


import sys
import os
curr_path = os.path.dirname(os.path.realpath('__file__'))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import torch
import datetime
import numpy as np
from common.utils import save_results, make_dir
from common.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    '''超参数
    '''

    def __init__(self):
        ################################## 环境超参数 ###################################
        self.algo_name = 'DQN'  # 算法名称
        self.env_name = 'CartPole-v1'  # 环境名称
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # 检测GPUgjgjlkhfsf风刀霜的撒发十
        self.seed = 10 # 随机种子，置0则不设置随机种子
        self.train_eps = 200  # 训练的回合数
        self.test_eps = 30  # 测试的回合数
        ################################################################################
        
        ################################## 算法超参数 ###################################
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.epsilon_start = 0.90  # e-greedy策略中初始epsilon
        self.epsilon_end = 0.01  # e-greedy策略中的终止epsilon
        self.epsilon_decay = 500  # e-greedy策略中epsilon的衰减率
        self.lr = 0.0001  # 学习率
        self.memory_capacity = 100000  # 经验回放的容量
        self.batch_size = 64  # mini-batch SGD中的批量大小
        self.target_update = 4  # 目标网络的更新频率
        self.hidden_dim = 128 #256  # 网络隐藏层
        ################################################################################

        ################################# 保存结果相关参数 ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True # 是否保存图片
        ################################################################################


def env_agent_config(cfg):
    ''' 创建环境和智能体
    '''
    env = gym.make(cfg.env_name)  # 创建环境
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n  # 动作维度
#     print(f'state dim: {env.observation_space}, action dim: {action_dim}')
    agent = DQN(state_dim, action_dim, cfg)  # 创建智能体
    if cfg.seed !=0: # 设置随机种子
        torch.manual_seed(cfg.seed)
        env.seed(cfg.seed)
        np.random.seed(cfg.seed)
    return env, agent


def train(cfg, env, agent):
    ''' 训练
    '''
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push(state, action, reward,
                              next_state, done)  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break
        if (i_ep + 1) % cfg.target_update == 0:  # 智能体目标网络更新
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, cfg.train_eps, ep_reward))
    print('完成训练！')
    env.close()
    return rewards, ma_rewards


def test(cfg, env, agent):
    print('开始测试!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    ############# 由于测试不需要使用epsilon-greedy策略，所以相应的值设置为0 ###############
    cfg.epsilon_start = 0.0  # e-greedy策略中初始epsilon
    cfg.epsilon_end = 0.0  # e-greedy策略中的终止epsilon
    ################################################################################
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        state = env.reset()  # 重置环境，返回初始状态
        while True:
            action = agent.choose_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.1f}")
    print('完成测试！')
    env.close()
    return rewards, ma_rewards


if __name__ == '__main__':
    cfg = Config()

    # env, agent = env_agent_config(cfg)
    # rewards, ma_rewards = train(cfg, env, agent)
    # make_dir(cfg.result_path, cfg.model_path)
    # agent.save(path=cfg.model_path)
    # save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    # plot_rewards(rewards, ma_rewards, cfg, tag='train')

    # tt = torch.tensor([[142.2, 124.2]])
    # print(tt.max(1)[1])

    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)  # 导入模型
    rewards, ma_rewards = test(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='test',
                 path=cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, cfg, tag="test")  # 画出结果