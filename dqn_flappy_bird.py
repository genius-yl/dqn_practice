import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import flappy_bird_gymnasium
from collections import deque

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 환경 설정
env = flappy_bird_gymnasium.make("FlappyBird-v0")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# DQN 모델 및 설정
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
memory = deque(maxlen=50000)
gamma = 0.99
batch_size = 32
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update = 10

# Epsilon-Greedy 방식으로 행동 선택
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    with torch.no_grad():
        return policy_net(torch.FloatTensor(state)).argmax().item()

# 경험을 저장
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# 모델 학습
def train():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 학습 실행
num_episodes = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        store_experience(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        train()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")

env.close()
torch.save(policy_net.state_dict(), "flappy_dqn.pth")
