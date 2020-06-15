import argparse
import gym
import torch
import numpy as np
from box_world_env import BoxWorld
from itertools import count
from agent.BaselineQLearningAgent import BaselineQLearningAgent
from agent.RelationalQLearningAgent import RelationalQLearningAgent

import torchvision.transforms as transforms

def generate_env_set(seeds, n, goal_lengths, num_distractors, distractor_length):
    for seed in seeds:
        env = BoxWorld(n, np.random.choice(goal_lengths), np.random.choice(num_distractors), distractor_length)
        env.seed(seed)
        yield env


def boxworld_state_to_tensor(x):
    return transforms.functional.to_tensor(x).unsqueeze(0).double()

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = BoxWorld(6, 2, 1, 1)
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

    episodes = 1000
    episode_lengths = []
    episode_solved = []
    target_update = 10

    # for episode in range(episodes):
    try:
        for episode, env in enumerate(generate_env_set(range(episodes), 6, [2], [0, 1, 2], 1)):
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
                    print(episode, t + 1, reward, meta, done)
                    episode_lengths.append(meta['episode']['length'])
                    episode_solved.append(meta['episode']['solved'])
                    break
                if not episode % target_update:
                    agent.update_target()
                print(episode, t, reward, meta, done)
    except KeyboardInterrupt:
        pass
    print(episode_lengths)
    import pickle
    pickle.dump({
        'lengths': episode_lengths,
        'solved': episode_solved
        }, open('rel_eps.pkl', 'wb'))
    torch.save(agent, 'agent_rel.pth')
    import matplotlib.pyplot as plt
    plt.plot(episode_lengths)
    plt.savefig('foo-rel.png')

if __name__ == '__main__':
    main()