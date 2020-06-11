import argparse
import gym
import torch
from box_world_env import BoxWorld
from itertools import count
from agent.BaselineQLearningAgent import BaselineQLearningAgent
from agent.RelationalQLearningAgent import RelationalQLearningAgent

import torchvision.transforms as transforms

def boxworld_state_to_tensor(x):
    return transforms.functional.to_tensor(x).unsqueeze(0).double()

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = BoxWorld(6, 4, 2, 2)
    agent = RelationalQLearningAgent(
        n_actions=env.action_space.n,
        gamma=0.99,
        batch_size=128,
        eps=0.9,
        eps_min=0.05,
        eps_decay=1e8,
        memory_capacity=1024,
        device=device,
        n_relational=2
    )
    env.reset()

    episodes = 100
    episode_lengths = []
    target_update = 10

    for episode in range(episodes):
        state = env.reset()
        state = boxworld_state_to_tensor(state).to(device)
        for t in count():
            action = agent.action(state)
            next_state, reward, done, meta = env.step(action.item())
            next_state = boxworld_state_to_tensor(next_state).to(device)
            reward = torch.Tensor([reward]).to(device)
            agent.push_transition(state, action, next_state, reward)
            state = next_state
            agent.optimize()
            if done:
                print(episode, t + 1)
                episode_lengths.append(t + 1)
                break
            if not episode % target_update:
                agent.update_target()
            print(episode, t, reward, meta, done)

if __name__ == '__main__':
    main()