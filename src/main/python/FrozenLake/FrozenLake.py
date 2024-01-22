import numpy as np
import gym
import random
import time
from IPython.display import clear_output

##################################
# Q-Learning Algorithm
##################################

env = gym.make("FrozenLake-v1")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 1000     # number of episodes agent is allowed to play when training
max_steps_per_episode = 100     # number of steps agent is allowed per episode

learning_rate = 0.1
discount_rate = 0.99

# Exploration-exploitation tradeoff variables
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# List to hold all the rewards from each episode
rewards_all_episodes = []

# Q-Learning algorithm
# First for loop contains everything that will happen for each episode
for episode in range(num_episodes):
    state = env.reset()     # reset environment

    terminated = False
    truncated = False
    rewards_current_episode = 0

    # Second for loop contains what will happen for each time step within the episode
    for step in range(max_steps_per_episode):

        # Exploration-exploitation tradeoff
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])    # find index of the max q value in array (exploit)
        else:
            action = env.action_space.sample()      # sample action randomly (explore)

        new_state, reward, terminated, truncated, info = env.step(action)    # take action with step, returns tuple

        # Update Q-table for Q(s,a)
        # q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
        #     learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + num_episodes * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state
        rewards_current_episode += reward

        if terminated:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.spit(np.array(rewards_all_episodes), num_episodes/1000)
    count = 1000
    print("****Average reward per thousand episodes****\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    # Print updated Q-table
    print("\n\n****Q-table****\n")
    print(q_table)



