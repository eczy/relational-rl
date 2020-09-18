import argparse
import os
import pickle
import shutil
from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from boxworld import BoxWorld
from util import boxworld_state_to_tensor, generate_env_set, mkdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("output_dir", metavar="output-dir", type=str)
    parser.add_argument("-f", action="store_true")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--n-env", type=int, default=100)
    parser.add_argument("--env-dim", type=int, default=6)
    parser.add_argument("--goal-lengths", type=tuple, nargs="+", default=(3,))
    parser.add_argument("--n-distractors", type=tuple, nargs="+", default=(0, 1, 2))
    parser.add_argument("--distractor-length", type=int, default=1)
    args = parser.parse_args()

    mkdir(args.output_dir, args.f)

    agent = torch.load(args.model)
    agent.policy.eval()
    agent.target.eval()

    envs = generate_env_set(
        range(args.n_env),
        args.env_dim,
        args.goal_lengths,
        args.n_distractors,
        args.distractor_length,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episode_lengths = []
    episode_solved = []

    try:
        for episode, env in enumerate(envs):
            state = boxworld_state_to_tensor(env.reset()).to(device)
            for t in count():
                action = agent.action(state)
                next_state, reward, done, meta = env.step(action.item())
                next_state = boxworld_state_to_tensor(next_state).to(device)
                reward = torch.Tensor([reward]).to(device)
                state = next_state
                if done:
                    print(episode, t + 1, reward, meta, done)
                    episode_lengths.append(meta["episode"]["length"])
                    episode_solved.append(meta["episode"]["solved"])
                    break
                print(episode, t, reward, meta, done)
    except KeyboardInterrupt:
        pass
    df = pd.DataFrame({"length": episode_lengths, "solved": episode_solved})
    df.to_csv(os.path.join(args.output_dir, "eval_results.csv"), index=False)


if __name__ == "__main__":
    main()
