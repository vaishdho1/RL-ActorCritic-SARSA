import numpy as np
from math import pi, exp, cos
import random
import math
import matplotlib.pyplot as plt
from itertools import product
import gym
import itertools
n = 25
k = 1
directions = [(0,-1),(-1,0),(0,1),(1,0)]
actionMap = {0:'L',1:'U',2:'R',3:'D'}
actions = ['L','U','R','D']
blocked_states = {12,17}
terminal_states = {24}
waterstate = 22 #(2+5*4)
goalstate = 24 #(4+4*5)
states = []
for i in range(25):
    if i!=12 and i!=17:
        states.append(i)
#blocked states
#Reward function:
reward = dict()
for i in range(n):
    for j in range(n):
        state = j+i*n
        reward[state] = {}
        for a in actions:
            reward[state][a] = {}
            for k in range(n):
                for l in range(n):
                    nxt = l+n*k
                    reward[state][a][nxt] = 0
            reward[state][a][goalstate] = 10
            reward[state][a][waterstate] = -10
            #Q3
            #reward[state][a][2] = 5

#Transition function
print(next)
nextState_action = dict()
for i in range(n):
    for j in range(n):
        state = j + i * n
        nextState_action[state] = {}
        for action in actions:
            nextState_action[state][action] = {}

for i in range(n):
    for j in range(n):
        state = j+i*n
        for ind,dir in enumerate(directions):
            #Go to the intended direction
            x = i+dir[0]
            y = j + dir[1]
            nextstate = y+x*n
            if state not in nextState_action[state][actions[ind]]:
                nextState_action[state][actions[ind]][state] = 0

            if x <0 or x>=n or y<0 or y>=n or (x==2 and y==2) or (x==3 and y==2):
                nextState_action[state][actions[ind]][state] += .8
            else:
                 if nextstate not in nextState_action[state][actions[ind]]:
                    nextState_action[state][actions[ind]][nextstate] = 0
                 nextState_action[state][actions[ind]][nextstate] += .8

            #Veer to the right
            x_r = i+directions[(ind+1)%4][0]
            y_r = j + directions[(ind+1)%4][1]
            nextstate_r = y_r+x_r*n


            if x_r <0 or x_r>=n or y_r<0 or y_r>=n or (x_r==2 and y_r==2) or (x_r==3 and y_r==2):
                nextState_action[state][actions[ind]][state] += .05
            else:
                if nextstate_r not in nextState_action[state][actions[ind]]:
                    nextState_action[state][actions[ind]][nextstate_r] = 0
                nextState_action[state][actions[ind]][nextstate_r] += .05
            #Veer to the left
            x_l = i+directions[ind-1][0]
            y_l = j + directions[ind-1][1]
            nextstate_l = y_l+x_l*n


            if x_l <0 or x_l>=n or y_l<0 or y_l>=n or (x_l==2 and y_l==2) or (x_l==3 and y_l==2):
                nextState_action[state][actions[ind]][state] += .05
            else:
                 if nextstate_l not in nextState_action[state][actions[ind]]:
                    nextState_action[state][actions[ind]][nextstate_l] = 0
                 nextState_action[state][actions[ind]][nextstate_l] += .05

            #Break down and staying in the same state
            nextState_action[state][actions[ind]][state] += .1

discount_parameter = 1
actions = [0,1,2,3]
numactions = 2
order = 1
states_val = 2

print(nextState_action.keys())

def find_fourier_series(combinations,state):
    #normalisedState = normalized_state(state)
    #print('norm',normalisedState)
    stateFeatures = [np.dot(comb,state) for comb in combinations]
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
    #print('policy_weights',policy_weights)
    vals = [np.dot(policy_weights[i], policy_features) for i in range(len(policy_weights))]
    return vals

def get_softmax(policy_value,sigma):
    #print('policy_value',policy_value)
    values = [sigma*v for v in policy_value]
    #print(values,'values')
    maxval = max(values)
    values = [math.exp(v-maxval) for v in policy_value]
    probs = [val/sum(values) for val in values]
    return probs

def find_valuefunction(v_weights, v_features):
    val = np.dot(v_weights, v_features)
    return val
'''
def normalized_state(state):
    #print('state',state)
    #print('ranges',ranges)
    res = [(state - ranges[0][0])/(ranges[0][1]-ranges[0][0])]
    #print('res',res)
    return res
'''

def find_action(policy_weights,features):
    policyvalues = find_policy_value(policy_weights, features)
    probs = get_softmax(policyvalues, sigma)
    #print('probs',probs)
    #print('actions',actions)
    action_index = random.choices(actions, probs)[0] # picked action
    action_prob = probs[action_index]
    return action_index,action_prob
def getNextstate(state,action):
    #print('state',state)
    #print('nextstate_action',nextState_action.keys())
    act = actionMap[action]
    next_states = list(nextState_action[state][act].keys())
    probs = list(nextState_action[state][act].values())
    nextstate = np.random.choice(next_states, p=probs)
    return nextstate
def run_episode(policy_weights, value_weights,combinations, policy_alpha, v_alpha, sigma):
    state = np.random.choice(states)
    gamma = 1
    done = False
    steps = 0
    var_I = 1
    totReward = 0
    while True:
        if state == goalstate:
            return policy_weights, value_weights, steps,totReward
        steps += 1
        #print('combinations',combinations)
        x = state/5
        y = state%5
        st = [x,y]
        features = find_fourier_series(combinations,st)
        #print('features',features)
        action_index,action_prob = find_action(policy_weights,features)
        next_state = getNextstate(state,action_index)
        r = reward[state][actionMap[action_index]][next_state]
        totReward += r
        next_features = find_fourier_series(combinations,st)
        nextstate_val = find_valuefunction(value_weights, next_features)
        curstate_val = find_valuefunction(value_weights, features)
        td_diff = r + (discount_parameter * nextstate_val) - curstate_val
        value_weights += float(v_alpha * td_diff) * features
        gradient_p = [-action_prob for _ in range(len(actions))]
        gradient_p[action_index] = 1-action_prob
        for i in range(len(actions)):
            policy_weights[i] += float(policy_alpha * td_diff * var_I) * gradient_p[i] * features
        var_I *= gamma
        state = next_state

def plot_episodes_action(overall_steps,numEpisodes):
    plt.title('Learning curve')
    plt.plot(overall_steps,[i for i in range(1, numEpisodes + 1)], overall_result,label='Mean',color='brown')
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

def plot_rewards(overall_reward,numEpisodes):
    plt.title('Performance Cartpole')
    plt.plot([i for i in range(1, numEpisodes + 1)], overall_reward,label='Mean',color='brown')
    #plt.errorbar([i for i in range(1, numEpisodes + 1)], overall_result, yerr=std_overall_result, linestyle='None', color='brown',label='Std Dev')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

numTimes = 1
numEpisodes = 500
alpha = 0.6
sigma = 1
order = 1
overall_result =[]
tot_steps = []
rewards = np.zeros((numTimes,numEpisodes))
for i in range(numTimes):
    result = []
    curLoop = []
    actions_count = []
    actions_taken = [0] * (numEpisodes + 1)
    policy_weights = get_policy_weights(states_val,order)
    value_weights = get_value_weights(states_val,order)
    combinations = find_state_combinations(order,states_val)
    reward_tot = []
    p_alpha = alpha
    v_alpha = alpha
    for j in range(numEpisodes):
        print(j)
        policy_weights, value_weights, timesteps,totReward= run_episode(policy_weights, value_weights,combinations, p_alpha, v_alpha, sigma)
        actions_count.append(timesteps)
        actions_taken[i+1] = actions_taken[i] + timesteps
        reward_tot.append(totReward)
        result.append(timesteps)
    rewards[i] = reward_tot
    tot_steps.append(actions_taken)
    overall_result.append(result)


overall_result = np.mean(overall_result, axis=0)
overall_steps = np.mean(tot_steps,axis = 0)
std_overall_result = np.std(overall_result, axis=0)
overall_reward = np.mean(rewards,axis = 0)



plot_rewards(overall_reward,numEpisodes)
