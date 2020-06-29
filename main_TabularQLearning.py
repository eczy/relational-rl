import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import sys
sys.path.insert(1, './../Box-World/')
import box_world_env


env_name = "boxplot"
n_rounds = num_episodes = 1000  # args.rounds
n_steps = 1000000  # args.steps
epsilon = 0.9  # should be 0.1 <= epsilon <= 1.0
# # decay_rate = 0.01
lr_rate = 0.81
# gamma = 0.96
gamma = 0.99

# env = box_world_env.BoxWorld(12, 4, 2, 2)
env = box_world_env.BoxWorld(6, 2, 1, 1)

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


episode_data = []


# policy func
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
      Creates an epsilon-greedy policy based
      on a given Q-function and epsilon.

      Returns a function that takes the state
      as an input and returns the probabilities
      for each action in the form of a numpy array
      of length of the action space(set of possible actions).
    """

    def policyFunction(state):
        Action_probabilities = np.ones(num_actions, dtype=float) * epsilon / num_actions

        # best_action = np.argmax(Q[state])
        # action = np.argmax(Q[state, :])
        best_action = max(Q[state.tostring()], key=Q[state.tostring()].get)

        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policyFunction


# Q-learning func
# TO-ASK: alpha = lr_rate? alpha = 0.6
def qLearning(env, num_episodes, epsilon, alpha=0.6, discount_factor=0.99):
    # discount_factor = 0.99/0.95/1.0
    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy
    """

    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    print(env.action_space.n)
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: {a: 0 for a in [0, 1, 2, 3]})
    # print(Q)
    # print(num_episodes)

    # Keeps track of useful statistics
    # old version:
    # stats = plotting.EpisodeStats(
    #     episode_lengths = np.zeros(num_episodes),
    #     episode_acc_rewards = np.zeros(num_episodes))

    stats = {"episode_lengths": np.zeros(num_episodes), "episode_acc_rewards": np.zeros(num_episodes)}

    episode_vs_length = {"episode_num": np.zeros(num_episodes), "episode_length": np.zeros(num_episodes)}

    episode_vs_acc_rewards = {"episode_num": np.zeros(num_episodes), "acc_rewards": np.zeros(num_episodes)}

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        for t in range(n_steps):

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to
            # the probability distribution
            action = np.random.choice(np.arange(
                len(action_probabilities)),
                p=action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done, info = env.step(action)
            # print(info)

            # Update statistics
            stats['episode_acc_rewards'][ith_episode] += reward
            stats['episode_lengths'][ith_episode] = t

            # Update data for episode_vs_length
            episode_vs_length['episode_num'][ith_episode] = ith_episode
            episode_vs_length['episode_length'][ith_episode] = t


            # print('==========================')
            # print(ith_episode)
            # print(reward)
            episode_vs_acc_rewards['episode_num'][ith_episode] = ith_episode
            episode_vs_acc_rewards['acc_rewards'][ith_episode] += reward
            # print('==========================')


            # TD Update
            # best_next_action = np.argmax(Q[next_state])
            # use try-except to avoid missing state error
            try:
                best_next_action = max(Q[next_state.tostring()], key=Q[next_state.tostring()].get)
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta
            except:
                pass
                #print(state, action)

            # done is True if episode terminated
            if done:
                episode_data.append((reward, done, info))
                break

            state = next_state

    return Q, stats, episode_vs_length, episode_vs_acc_rewards


Q, stats, episode_vs_length, episode_vs_acc_rewards = qLearning(env, num_episodes, epsilon)

# print("FINAL Q...")
# print(Q)

# figure #1 - episode_lengths vs. episode_accumulated_rewards
# print("FINAL STATS...")
# print(stats)
# fig, ax = plt.subplots()
# ax.plot(stats['episode_lengths'], stats['episode_acc_rewards'])
# fig.show()

# figure #2 - episode vs. episode_length
# print("FINAL EPISODE VS. LENGTHS...")
# print(episode_vs_length)
# fig_2, ax_2 = plt.subplots()
# ax_2.plot(episode_vs_length['episode_num'], episode_vs_length['episode_length'])
# fig_2.show()


# # figure #3 - episode vs accumulated rewards
# print("FINAL EPISODE VS. ACCUMULATED REWARDS...")
# print(episode_vs_acc_rewards)
# fig_3, ax_3 = plt.subplots()
# ax_3.plot(episode_vs_acc_rewards['episode_num'], episode_vs_acc_rewards['acc_rewards'])
# fig_3.show()



# # keep the figure stays for some time
# plt.show(block=False)
# plt.pause(10)
# plt.close()



# rliu modifciations to generate the graph:
# generate episode_length vs episode_solved:

stat1 = [
  (d,   # done
   i['episode']['solved'] if d else None ,  # solved
   i['episode']['r'] if d else None,        # reward
   i['episode']['length'] if d else None)   # num-steps taken
  for (r, d, i) in episode_data
]

episode_lengths = []
episode_solved = []
for (done, solved, reward, num_steps) in stat1:
    episode_lengths.append(num_steps)
    episode_solved.append(solved)
import pickle
pickle.dump({
    'lengths': episode_lengths,
    'solved': episode_solved
    }, open('tabqlearning_eps.pkl', 'wb'))



env.close()



