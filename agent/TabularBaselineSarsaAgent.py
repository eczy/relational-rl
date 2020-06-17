# import gym
import box_world_env
import time
import matplotlib.pyplot as plt
# import os
import numpy as np
from collections import defaultdict
import json

env_name = "boxplot"
n_rounds = 2000 # args.rounds
n_steps = 100000 # args.steps


# n, goal_length, num_distractor, distractor_length, seed=None
# env = box_world_env.BoxWorld(12, 4, 2, 2)
env = box_world_env.BoxWorld(6, 2, 1, 1)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


epsilon = 0.9
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01

lr_rate = 0.81
gamma = 0.96
# gamma = 0.99

# distractor length <-- keep low; maybe 1


Q = defaultdict(lambda: {a: 0 for a in [0,1,2,3]})

def choose_action(state):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        # action = np.argmax(Q[state, :])
        action = max(
            Q[json.dumps(state)], key=Q[json.dumps(state)].get
        )
    return action

def learn(state, state2, reward, action, action2):
    try:
        predict = Q[json.dumps(state)][action]
        target = reward + gamma * Q[json.dumps(state2)][action2]
        Q[json.dumps(state)][action] = Q[json.dumps(state)][action] + lr_rate * (target - predict)
    except:
        print(state, action)

run_data = []
state = None
pstate = None
episode_data = []


def run(ca):
    for i_episode in range(n_rounds):
        # print('Starting new game!')

        # state = env.reset()
        env.reset()
        # pstate = env.player_position
        # print(type(env.player_position.tolist()))
        # print(type(env.goal_position))
        state = [env.player_position.tolist(), env.goal_position, env.keys_info[env.keys_idx+1]]
        # print(state)
        action = ca(state)
        for t in range(n_steps):

            # time.sleep(0.2)
            # print(f"Episode {i_episode+1} and step {t+1}")
            # print(env.render(mode="rgb_array"))
            # env.render()

            _, reward, done, info = env.step(action)

            # pstate2 = env.player_position
            state2 = [env.player_position.tolist(), env.goal_position, env.keys_info[env.keys_idx+1]]
            action2 = ca(state2)

            learn(state, state2, reward, action, action2)

            state, action = state2, action2

            # print(ACTION_LOOKUP[action], reward, done, info)
            # print(len(state), len(state[0]), len(state[0][0]))

            if done:
                episode_data.append((reward, done, info))
                # print(f"Episode {i_episode} FINISHED after {t + 1} timesteps")
                # print(f"done={done} info={info}")

                # print(env.render(mode="rgb_array"))
                # print("solved", env.keys_info, "goal pos: ", env.goal_position)
                break
            elif t == (n_steps - 1):
                # raise Exception;
                episode_data.append((reward, done, info))
                # print("Episode timed out after {} timesteps".format(t+1))
                # print(f"Episode {i_episode} TIMED OUT after {t + 1} timesteps")
                # print(env.render(mode="rgb_array"))
                # print("NOT solved", env.keys_info, "goal pos: ", env.goal_position)
                break

run(choose_action)
run_data.append(episode_data)
# episode_data = []

stat1 = [
  (d,   # done
   i['episode']['solved'] if d else None ,  # solved
   i['episode']['r'] if d else None,        # reward
   i['episode']['length'] if d else None)   # num-steps taken
  for (r, d, i) in run_data[0]
]

stat2 = []
for r in range(n_rounds):
    fs = 0.0
    for i in range(r):
        if stat1[i][1] == True:
            fs += 1.0
    fs = fs / (r + 1)
    stat2.append(fs)

# total number of episodes that correctly solved
stat3 = sum(map(lambda k : k[1] == True, stat1))


env.close()

# fig, ax = plt.subplots()
# ax.plot(stat2)
# fig.show()
