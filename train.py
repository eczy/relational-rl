import argparse
import os
import pickle
from itertools import count

import gym
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from agent import BaselineQLearningAgent, RelationalQLearningAgent
from boxworld import BoxWorld
from util import boxworld_state_to_tensor, generate_env_set, mkdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, choices=["baseline", "relational"])
    parser.add_argument("output_dir", metavar="output-dir", type=str)
    parser.add_argument("-f", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon-max", type=float, default=0.9)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=1e8)
    parser.add_argument("--memory-size", type=int, default=1028)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--n-env", type=int, default=100)
    parser.add_argument("--env-dim", type=int, default=6)
    parser.add_argument("--goal-lengths", type=tuple, nargs="+", default=(3,))
    parser.add_argument("--n-distractors", type=tuple, nargs="+", default=(0, 1, 2))
    parser.add_argument("--distractor-length", type=int, default=1)
    args = parser.parse_args()

    mkdir(args.output_dir, overwrite=args.f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.agent == "baseline":
        agent = BaselineQLearningAgent(
            n_actions=4,
            gamma=args.gamma,
            batch_size=args.batch_size,
            eps=args.epsilon_max,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            memory_capacity=args.memory_size,
            device=device,
            n_attention_blocks=6,
        )
    elif args.agent == "relational":
        agent = RelationalQLearningAgent(
            n_actions=4,
            gamma=args.gamma,
            batch_size=args.batch_size,
            eps=args.epsilon_max,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            memory_capacity=args.memory_size,
            device=device,
            n_attention_blocks=2,
        )

    envs = generate_env_set(
        range(args.episodes),
        args.env_dim,
        args.goal_lengths,
        args.n_distractors,
        args.distractor_length,
    )

    episode_lengths = []
    episode_solved = []

    try:
        for episode, env in enumerate(envs):
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
                    episode_lengths.append(meta["episode"]["length"])
                    episode_solved.append(meta["episode"]["solved"])
                    break
                if not episode % args.target_update:
                    agent.update_target()
                print(episode, t, reward, meta, done)
    except KeyboardInterrupt:
        pass
    df = pd.DataFrame({"length": episode_lengths, "solved": episode_solved})
    df.to_csv(os.path.join(args.output_dir, "train_results.csv"), index=False)
    torch.save(agent, os.path.join(args.output_dir, f"{args.agent}_agent.pt"))


if __name__ == "__main__":
    main()
