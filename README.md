 # RL-Algorithms: Actor-Critic & n-step SARSA
## Introduction
This project implements two reinforcement learning algorithms: One-step Actor-Critic and Episodic Semi-Gradient n-step SARSA. The algorithms are tested on several environments from OpenAI's Gym library, including Cartpole and Acrobot, as well as a custom environment, 687-Gridworld.

## Algorithms

### One-step Actor-Critic
The One-step Actor-Critic algorithm combines value-based and policy-based methods by using an actor to choose actions and a critic to evaluate them. This allows for more stable learning and improved performance in continuous action spaces.

### Episodic Semi-Gradient n-step SARSA
Episodic Semi-Gradient n-step SARSA is a temporal difference learning method that updates action-value estimates using multiple steps of experience within an episode. This approach can yield more accurate value estimates and faster learning compared to single-step methods.

## Environments

### Cartpole
The Cartpole environment is a classic control problem where the goal is to balance a pole on a cart by applying forces to the cart. The episode ends when the pole falls over or the cart moves out of bounds.

### Acrobot
The Acrobot environment involves a two-link robotic arm where the goal is to swing the lower link up to a certain height. This is a more complex control problem requiring precise coordination of the links.

### 687-Gridworld
The 687-Gridworld is a custom grid-based environment where the agent must navigate from a start state to a goal state while avoiding obstacles. The environment provides a simple yet effective testbed for evaluating RL algorithms.

## Implementation Details
The implementation of both algorithms follows the standard reinforcement learning framework. Here are some key points:

- **Actor-Critic**: The actor network outputs a probability distribution over actions, while the critic network estimates the value of the current state.
- **n-step SARSA**: The algorithm updates Q-values based on the cumulative reward of n steps into the future, combining elements of both SARSA and Monte Carlo methods
