from torchvision import transforms
from itertools import count
import pickle
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import seaborn as sns
import numpy as np
from box_world_env import BoxWorld

def boxworld_state_to_tensor(x):
    return transforms.functional.to_tensor(x).unsqueeze(0).double()


def generate_env_set(seeds, n, goal_lengths, num_distractors, distractor_length):
    for seed in seeds:
        env = BoxWorld(n, np.random.choice(goal_lengths), np.random.choice(num_distractors), distractor_length)
        env.seed(seed)
        yield env

import torch
agent_base = torch.load('agent_base.pth')
agent_rel = torch.load('agent_rel.pth')

agent_base.policy.eval()
agent_base.target.eval()
agent_rel.policy.eval()
agent_rel.target.eval()

len_3_env = generate_env_set(range(100), 6, [3], [0, 1, 2], 1)

len_4_env = generate_env_set(range(100, 200), 6, [4], [0, 1, 2], 1)

episodes = 1000
episode_lengths = []
episode_solved = []
target_update = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = agent_base

# for episode in range(episodes):
try:
    for episode, env in enumerate(len_3_env):
        state = env.reset()
        state = boxworld_state_to_tensor(state).to(device)
        for t in count():
            action = agent.action(state)
            next_state, reward, done, meta = env.step(action.item())
            next_state = boxworld_state_to_tensor(next_state).to(device)
            reward = torch.Tensor([reward]).to(device)
            state = next_state
            if done:
                print(episode, t + 1, reward, meta, done)
                episode_lengths.append(meta['episode']['length'])
                episode_solved.append(meta['episode']['solved'])
                break
            print(episode, t, reward, meta, done)
except KeyboardInterrupt:
    pass
print(episode_lengths)
import pickle
pickle.dump({
    'lengths': episode_lengths,
    'solved': episode_solved
    }, open('base_len_3_eps.pkl', 'wb'))


episodes = 1000
episode_lengths = []
episode_solved = []
target_update = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = agent_base

# for episode in range(episodes):
try:
    for episode, env in enumerate(len_4_env):
        state = env.reset()
        state = boxworld_state_to_tensor(state).to(device)
        for t in count():
            action = agent.action(state)
            next_state, reward, done, meta = env.step(action.item())
            next_state = boxworld_state_to_tensor(next_state).to(device)
            reward = torch.Tensor([reward]).to(device)
            state = next_state
            if done:
                print(episode, t + 1, reward, meta, done)
                episode_lengths.append(meta['episode']['length'])
                episode_solved.append(meta['episode']['solved'])
                break
            print(episode, t, reward, meta, done)
except KeyboardInterrupt:
    pass
print(episode_lengths)
import pickle
pickle.dump({
    'lengths': episode_lengths,
    'solved': episode_solved
    }, open('base_len_4_eps.pkl', 'wb'))
