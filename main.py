# Class for actor network
# Class for critic network
# Class for action noise
# Class for replay buffer
# Class for agent functionality
    # Memory, actor/critic networks, target nets, tau

# Initialize actor & critic networks 𝛍(s|𝛉^𝛍) Q(s, a|𝛉^Q)

# Initialize target networks with weights from main networks

# Initialize replay buffer R

# For a large number of episodes
    # Initialize noise process N for action exploration
    # Reset environment
    # For each step of game
        # at = 𝛍(st|𝛉^𝛍) + Nt
        # Take action at and get new state, reward
        # Store (st, at, rt, St+1, donet) in R
        # Sample random minibatch of (si, ai, ri, si+1) from R
        # yi = ri + rQ'(si+1, 𝛍'(si+1|𝛉^𝛍')|𝛉^Q)
        # Update critic by minimizing :=1/N𝚺(yi-Q(si, ai|𝛉^Q))^2
        # Update actor 𝛁𝛉≈1/N𝚺𝛁𝛉Q(s, a|𝛉^Q)|s=s, a=𝛍(st|𝛉^𝛍)
        # 𝛉^Q <- 𝛕𝛉^Q+(1-𝛄)𝛉^Q
        # 𝛉^𝛍 <- 𝛕𝛉^𝛍 + (1-𝛄)𝛉^𝛉^𝛍

import numpy as np
import gym
from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001, \
                  batch_size=64, fc1_dims=400, fc2_dims=300, n_actions=env.action_space.shape[0])
    n_games =1000
    filename = 'LunarLander_alpha_' + str(agent.alpha) + '_beta_' + \
        str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = '/Users/soankim/PycharmProjects/ReinforcementLearning/' + filename + '.png'
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)