import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np


# 策略网络：参数 θ 对应网络的权重和偏置
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        # 这里的线性层权重和偏置就是公式中的参数θ
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出logits，未经过softmax的原始分数
        x = self.fc2(x)
        return x

    def get_action_and_logprob(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        # 将logits转换为概率分布: π_θ(a|s)
        probs = F.softmax(logits, dim=-1)
        # 从π_θ(a|s)中采样一个动作
        dist = Categorical(probs)
        action = dist.sample()
        # 计算log π_θ(a|s)
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class REINFORCE:
    def __init__(self, env, lr=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma

        # 初始化策略网络及其优化器
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []

    def collect_trajectory(self):
        state = self.env.reset()
        done = False

        # 清空轨迹
        self.log_probs = []
        self.rewards = []

        while not done:
            # 从策略网络中选取动作并获取log概率
            action, log_prob = self.policy.get_action_and_logprob(state)

            # 执行动作
            next_state, reward, done, _ = self.env.step(action)

            # 保存log概率和奖励
            self.log_probs.append(log_prob)
            self.rewards.append(reward)

            state = next_state

    def compute_returns(self):
        # 计算G_t，从t时刻开始的未来折扣回报
        returns = []
        G = 0

        # 逆序计算未来回报
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # 可选：归一化回报以减少方差
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        return returns

    def update_policy(self):
        # 计算所有时间步的G_t值
        returns = self.compute_returns()

        # 策略更新
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            # 对应公式：-log π_θ(a_t|s_t) · G_t
            # 负号是因为PyTorch执行梯度下降，而我们需要梯度上升
            policy_loss.append(-log_prob * G)

        # 累积所有时间步的损失
        policy_loss = torch.cat(policy_loss).sum()

        # 梯度清零
        self.optimizer.zero_grad()
        # 反向传播，计算∇_θ log π_θ(a_t|s_t) · G_t
        policy_loss.backward()
        # 参数更新：θ ← θ + α · ∇_θ log π_θ(a_t|s_t) · G_t
        self.optimizer.step()

    def train(self, n_episodes):
        for episode in range(n_episodes):
            # 收集一条轨迹
            self.collect_trajectory()
            # 更新策略参数
            self.update_policy()

            episode_reward = sum(self.rewards)
            print(f"Episode {episode}: Total reward: {episode_reward}")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np


# 策略网络：参数化π_θ(a|s)
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action_and_logprob(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# 价值网络：参数化V_w(s)作为基线
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        # 这里的参数w对应价值网络的权重和偏置
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 输出状态价值V_w(s)
        x = self.fc2(x)
        return x









class REINFORCEWithBaseline:
    def __init__(self, env, policy_lr=0.01, value_lr=0.01, gamma=0.99):
        self.env = env
        self.gamma = gamma

        # 初始化策略网络(Actor)和优化器
        self.policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        # 初始化价值网络(Critic)和优化器
        self.value = ValueNetwork(env.observation_space.shape[0])
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

        self.log_probs = []
        self.rewards = []
        self.states = []

    def collect_trajectory(self):
        state = self.env.reset()
        done = False

        self.log_probs = []
        self.rewards = []
        self.states = []

        while not done:
            self.states.append(state)

            action, log_prob = self.policy.get_action_and_logprob(state)
            next_state, reward, done, _ = self.env.step(action)

            self.log_probs.append(log_prob)
            self.rewards.append(reward)

            state = next_state

    def compute_returns(self):
        returns = []
        G = 0

        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        return returns

    def update_networks(self):
        # 计算G_t
        returns = self.compute_returns()

        # 计算所有状态的价值估计V_w(s_t)
        states = torch.FloatTensor(np.array(self.states))
        values = self.value(states).squeeze()

        # 计算优势：δ_t = G_t - V_w(s_t)
        advantages = returns - values.detach()  # 分离，不要对其求导

        # 策略损失：-∑ log π_θ(a_t|s_t) · δ_t
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)  # 负号用于梯度上升
        policy_loss = torch.cat(policy_loss).sum()

        # 价值损失：1/2 · ∑ (G_t - V_w(s_t))²
        value_loss = 0.5 * ((returns - values) ** 2).sum()

        # 更新策略网络参数 θ
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新价值网络参数 w
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, n_episodes):
        for episode in range(n_episodes):
            self.collect_trajectory()
            self.update_networks()

            episode_reward = sum(self.rewards)
            print(f"Episode {episode}: Total reward: {episode_reward}")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np


# Actor网络：策略π_θ(a|s)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_action_and_logprob(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


# Critic网络：状态价值函数V_w(s)
class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ActorCritic:
    def __init__(self, env, actor_lr=0.001, critic_lr=0.005, gamma=0.99):
        self.env = env
        self.gamma = gamma

        # 初始化Actor网络和优化器
        self.actor = ActorNetwork(env.observation_space.shape[0], env.action_space.n)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 初始化Critic网络和优化器
        self.critic = CriticNetwork(env.observation_space.shape[0])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train_one_episode(self):
        state = self.env.reset()
        done = False
        episode_reward = 0

        while not done:
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state)

            # 从策略中采样动作
            action, log_prob = self.actor.get_action_and_logprob(state)

            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            # 获取当前状态的价值估计V_w(s)
            value = self.critic(state_tensor)

            # 如果游戏结束，下一状态价值为0，否则计算下一状态价值V_w(s')
            if done:
                next_value = 0
            else:
                next_state_tensor = torch.FloatTensor(next_state)
                next_value = self.critic(next_state_tensor).detach()  # 不计算梯度

            # 计算TD误差: δ_t = r_t + γV_w(s_{t+1}) - V_w(s_t)
            # 这是优势估计的一种形式
            delta = reward + self.gamma * next_value - value

            # 计算Critic(值函数)损失: 1/2 · δ_t²
            critic_loss = 0.5 * delta ** 2

            # 计算Actor(策略)损失: -log π_θ(a_t|s_t) · δ_t
            # 注意负号：PyTorch默认执行梯度下降，但我们需要梯度上升
            actor_loss = -log_prob * delta.detach()  # 分离delta，不对其求导

            # 更新Critic网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 更新Actor网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 移动到下一状态
            state = next_state

        return episode_reward

    def train(self, num_episodes):
        for episode in range(num_episodes):
            reward = self.train_one_episode()
            print(f"Episode {episode}: Total reward: {reward}")
