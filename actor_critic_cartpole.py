import numpy as np
from math import pi, exp, cos
import random
import math
import matplotlib.pyplot as plt
from itertools import product
import gym
import itertools
discount_parameter = 1
actions = [0, 1]
numactions = 2
state_variables = ['x', 'v', 'angle', 'angular_velocity']
v_max_val = 4
ang_v_max_val = 2.5
ranges = [[-4.8, 4.8],[-4, 4],[-0.418, 0.418],[-2.5, 2.5]]
n = 4
k=1

def find_fourier_series(combinations,state):
    normalisedState = normalized_state(state)
    stateFeatures = [np.dot(comb,normalisedState) for comb in combinations]
    stateFeatures = [np.cos(pi*value) for value in stateFeatures]

    return np.array(stateFeatures)
def find_state_combinations(k,n):
    combinations =list(itertools.product(range(k+1),repeat=n))
    combinations = [val for val in combinations]
    return np.array(combinations)

def get_value_weights(n,k):
    weights_array = np.array([0.0 for _ in range((k+1)**n)])
    return weights_array

def get_policy_weights(n,k):
     weights_array = [np.array([0.0 for _ in range((k+1)**n)]) for _ in range(len(actions))]
     return weights_array

def find_policy_value(policy_weights,policy_features):
    vals = [np.dot(policy_weights[i], policy_features) for i in range(len(policy_weights))]
    return vals

def get_softmax(policy_value,sigma):
    values = [sigma*v for v in policy_value]
    maxval = max(values)
    values = [math.exp(v-maxval) for v in policy_value]
    probs = [val/sum(values) for val in values]
    return probs

def find_valuefunction(v_weights, v_features):
    val = np.dot(v_weights, v_features)
    return val

def normalized_state(state):
    res = []
    for i in range(n):
        res.append((state[i] - ranges[i][0]) /(ranges[i][1]-ranges[i][0]))
    return res

def find_action(policy_weights,features):
    policyvalues = find_policy_value(policy_weights, features)
    probs = get_softmax(policyvalues, sigma)
    action_index = random.choices(actions, probs)[0] # picked action
    action_prob = probs[action_index]
    return action_index,action_prob

def run_episode(policy_weights, value_weights,combinations, policy_alpha, value_alpha, sigma):
    env = gym.make('CartPole-v1')
    state = env.reset()[0]
    gamma = 1
    done = False
    steps = 0
    var_I = 1
    totReward = 0
    while True:
        if done or steps == 500:
            return policy_weights, value_weights, steps,totReward
        steps += 1
        features = find_fourier_series(combinations,state)
        action_index,action_prob = find_action(policy_weights,features)
        next_state, reward, done, _, _ = env.step(action_index)
        totReward += reward
        next_features = find_fourier_series(combinations,next_state)
        nextstate_val = find_valuefunction(value_weights, next_features)
        curstate_val = find_valuefunction(value_weights, features)
        td_diff = reward + (discount_parameter * nextstate_val) - curstate_val
        value_weights += float(value_alpha * td_diff) * features
        gradient_p = [-action_prob for _ in range(len(actions))]
        gradient_p[action_index] = 1-action_prob
        for i in range(len(actions)):
            policy_weights[i] += float(policy_alpha * td_diff * var_I) * gradient_p[i] * features
        var_I *= gamma
        state = next_state

def plot_episodes_action(overall_steps,numEpisodes):
    plt.title('Learning curve')
    plt.plot(overall_steps,[i for i in range(1, numEpisodes + 1)], numEpisodes,label='Mean',color='green')
    #plt.errorbar([i for i in range(1, numEpisodes + 1)], overall_result, yerr=std_overall_result, linestyle='None', color='brown',label='Std Dev')
    plt.xlabel('Number of steps')
    plt.ylabel('Epsiodes')
    plt.show()

def plot_performance(overall_result,numEpisodes):
    plt.title('Learning curve Cartpole')
    plt.plot([i for i in range(1, numEpisodes + 1)], overall_result,label='Mean',color='brown')
    #plt.errorbar([i for i in range(1, numEpisodes + 1)], overall_result, yerr=std_overall_result, linestyle='None', color='brown',label='Std Dev')
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps')
    plt.show()

def plot_rewards(overall_reward,std_deviation,numEpisodes):
    plt.title('Performance Cartpole')
    #plt.plot([i for i in range(1, numEpisodes + 1)], overall_reward,label='Mean',color='brown')
    plt.plot([i for i in range(1, numEpisodes + 1)], overall_reward,label='Mean',color='green')
    plt.fill_between([i for i in range(1, numEpisodes + 1)],overall_reward - std_deviation, overall_reward + std_deviation, alpha=0.2, color='green')
    #plt.errorbar([i for i in range(1, numEpisodes + 1)], overall_reward, yerr=std_deviation, linestyle='None', color='blue',label='Std Dev')
    #plt.errorbar([i for i in range(1, numEpisodes + 1)], overall_result, yerr=std_overall_result, linestyle='None', color='brown',label='Std Dev')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

numTimes = 3
numEpisodes = 600
alphas = .01
policy_alpha = 0.01
value_alpha = 0.01
sigma = 1
order = 1
overall_result =[]
tot_steps = []
rewards = np.zeros((numTimes,numEpisodes))


overall_rewards = []
overall_actions = []
for i in range(numTimes):
    reward_tot = []
    tot_actions = 0
    cum_actions = []
    policy_weights = get_policy_weights(n,k)
    value_weights = get_value_weights(n,k)
    combinations = find_state_combinations(k,n)
    for j in range(numEpisodes):
        policy_weights, value_weights,timesteps,totReward = run_episode(policy_weights, value_weights,combinations,policy_alpha, value_alpha, sigma)
        tot_actions += timesteps
        cum_actions.append(tot_actions)
        reward_tot.append(totReward)
    overall_actions.append(cum_actions)
    overall_rewards.append(reward_tot)

avg_rewards = np.mean(overall_rewards, axis=0)
std_rewards = np.std(overall_rewards, axis=0)
avg_actions = np.mean(overall_actions, axis=0)

plot_episodes_action(avg_actions,numEpisodes)
plot_rewards(avg_rewards,std_rewards,numEpisodes)
