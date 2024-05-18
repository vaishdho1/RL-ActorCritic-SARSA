

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from itertools import product

def select_state(next_states, transition_probabilities):
    state_strings = [str(state) for state in next_states]
    selected_state_str = np.random.choice(state_strings, p=transition_probabilities)
    selected_state = tuple(map(int, selected_state_str.strip('[]()').replace(',', '').split()))
    return selected_state

def epsilon_greedy_action(action_value, eps):
    if np.random.random() < eps:
        action = np.random.choice(len(action_value))
    else:
        action = np.argmax(action_value)
    return action

def compute_fourier_basis(order, num_state_vars):
    combinations = list(product(*(range(order + 1) for _ in range(num_state_vars))))
    return np.array(combinations)

def compute_action_value(weights, features, normalized_state):
    phi = compute_phi(features, normalized_state)
    action_value = np.dot(weights, phi)
    return action_value[0]  

def compute_phi(features, normalized_state):
    phi = np.cos( pi * np.dot(features, np.array(list(normalized_state)).reshape(len(normalized_state), 1)))
    return phi

def get_reward(state):
    if state == (4, 4):
        return 10
    elif state == water:
        return -10
    else:
        return 0

def check_state(state, new_state):
    row, col = new_state
    if new_state in obstacles or row < 0 or col < 0 or row >= grid_size[0] or col >= grid_size[1]:
        return state
    else:
        return new_state

def compute_next_states(state, action):
    row, col = state
    new_states = []

    def move(direction):
        new_states.append(check_state(state, direction))

    if action == 'AU':
        move((row - 1, col))
        move((row, col + 1))
        move((row, col - 1))

    elif action == 'AD':
        move((row + 1, col))
        move((row, col + 1))
        move((row, col - 1))

    elif action == 'AL':
        move((row, col - 1))
        move((row - 1, col))
        move((row + 1, col))

    elif action == 'AR':
        move((row, col + 1))
        move((row + 1, col))
        move((row - 1, col))

    new_states.append(state)
    return new_states

def run_episode(weights, features, alpha, eps, n_steps):
    T = 500
    t = 0

    done = False
    state = [0, 0]
    vals = []
    for i in range(len(actions)):
        vals.append(compute_action_value(weights[i], features, state))
    action_state_value = vals
    action = epsilon_greedy_action(action_state_value, eps)

    states_memory = [state]
    action_memory = [action]
    reward_memory = []
    total_reward=0
    
    while state!=(4, 4):
        state = states_memory[-1]
        action = action_memory[-1]

        possible_next_states = compute_next_states(state, actions[action])
        next_state = select_state(possible_next_states, transition_probabilities)

        states_memory.append(next_state)
        reward = get_reward(next_state)
        if reward==10:
            total_reward+=reward
            break
            
        total_reward+=reward
        reward_memory.append(reward)
        
        vals = []
        for i in range(len(actions)):
            vals.append(compute_action_value(weights[i], features, state))
        action_state_value = vals
        action = epsilon_greedy_action(action_state_value, eps)
        action_memory.append(action)

        tau = t - n_steps + 1
        if tau >= 0:
            G = sum([ pow(discount_parameter, i - tau - 1) * reward_memory[i] for i in range(tau+1, min(tau + n_steps, T))])
            if tau + n_steps < T: 
                G += pow(discount_parameter, n_steps) * compute_action_value(weights[action_memory[tau + n_steps]], features,states_memory[tau + n_steps])
            
            for i in range(len(actions)):
                weights[i] += float(alpha) * (G - compute_action_value(weights[action_memory[tau]], features, states_memory[tau])) * np.transpose(compute_phi(features ,states_memory[tau]))

        t += 1

    return weights, t, total_reward

actions = ['AU', 'AD', 'AL', 'AR']
state_variables = ['x', 'y']
num_state_variables = len(state_variables)
transition_probabilities = [0.8, 0.05, 0.05, 0.1]
grid_size = (5, 5)
obstacles = [(2, 2), (3, 2)]
water = (4, 2)
goal_state = (4, 4)

discount_parameter = 1
threshold_delta = 0.0001 

numTimes = 1
numEpisodes = 1000
alpha = 0.001
epsilon = 0.9
n_steps = 1
order = 1
decay_epsilon_param = 0.1

tot_res = []
cumulative_rewards = []  

for i in range(numTimes):
    decay_epsilon = epsilon
    reward_list = []
    cumulative_reward = 0  
    total = int(pow(order + 1, len(state_variables)))
    weights = [np.array([0.0] * total).reshape(1, total) for i in actions]
    combinations = compute_fourier_basis(order, num_state_variables)
    
    for i in range(numEpisodes):
        if (i + 1) % 50 == 0 and decay_epsilon > 0:
            decay_epsilon -= decay_epsilon_param
            decay_epsilon = max(0, decay_epsilon)

        weights, timesteps, total_reward = run_episode(weights, combinations, alpha, decay_epsilon, n_steps)
        cumulative_reward += total_reward  
        reward_list.append(total_reward)
        cumulative_rewards.append(cumulative_reward)  

plt.title('Performance Gridworld')
plt.plot([i for i in range(1, 1 + numEpisodes)], reward_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()

plt.title('Cumulative Rewards Graph')
plt.plot([i for i in range(1, 1 + numEpisodes)], cumulative_rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Rewards')
plt.show()



