import os
import shutil

import numpy as np
import torchvision.transforms as transforms

from boxworld import BoxWorld


def generate_env_set(seeds, n, goal_lengths, num_distractors, distractor_length):
    for seed in seeds:
        env = BoxWorld(
            n,
            np.random.choice(goal_lengths),
            np.random.choice(num_distractors),
            distractor_length,
        )
        env.seed(seed)
        yield env


def boxworld_state_to_tensor(x):
    return transforms.functional.to_tensor(x).unsqueeze(0).double()


def mkdir(path, overwrite=False):
    if os.path.isdir(path) and not overwrite:
        raise FileExistsError
    elif os.path.isdir(path) and overwrite:
        shutil.rmtree(path)
    os.makedirs(path)
