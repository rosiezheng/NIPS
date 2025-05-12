#Rosie Zheng, s1087207
#NIPS Final Project
#May 1st, 2022

import matplotlib.pyplot as plt
import numpy as np
import random
import gym

#setup gym environment
env = gym.make("Taxi-v3").env

#create table with zeros to store state-action pairs
Q_table = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 30000 #number of training episodes: 10000 or 30000
show_trained_episodes = 100 #number of episodes run after training

#defining parameters
alpha = 0.1 #learning rate
gamma = 0.99 #discount factor

#epsilon-greedy action selection (exploration-exploitation)
#due to long running time and time shortage, I couldn't include the decay anymore unfortunately
epsilon = 0.1 #0.1, 0.5, or 0.8
# max_epsilon = 1
# min_epsilon = 0.01
# epsilon_decay_rate = 0.001

#plotting rewards
plot_all_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards = 0
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() #pick new action
        else:
            action = np.argmax(Q_table[state]) #pick action with highest reward in history

        next_state, reward, done, info = env.step(action)        
        old_value = Q_table[state, action]

        #update Q-table for Q(s,a) with learning rate
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * np.max(Q_table[next_state]))
        Q_table[state, action] = new_value

        state = next_state
        rewards += reward

    #due to long running time and time shortage, I couldn't include this anymore unfortunately
    # #epsilon rate decay
    # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
    # #learning rate decay
    # alpha = alpha / (alpha + episode)

    plot_all_rewards.append(rewards)
     
    if episode % 100 == 0:
        print(f"Completed episodes: {episode}")

print("Training done\n")

#evaluate and show agent's performance after Q-learning
epochs_total = 0

for _ in range(show_trained_episodes):
    state = env.reset()
    epochs = 0 
    done = False
    
    while not done:
        action = np.argmax(Q_table[state])
        state, reward, done, info = env.step(action)
        epochs += 1
        env.render()

        print(f"Epoch: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")

    epochs_total += epochs
    print(f"Total epochs: {epochs_total}")

print(f"Results after {show_trained_episodes} episodes:")
print(f"Average timesteps per episode: {epochs_total / show_trained_episodes}")

#updated Q-table
print("\nQ-table\n")
print(Q_table)

#plot reward over number of training episodes
plt.plot(np.arange(0, num_episodes, 1), plot_all_rewards)
plt.xlabel("training episodes")
plt.ylabel("reward")
plt.title("Reward over training episodes (n=30000, $\epsilon=0.1$)")
plt.show()