import torch as T
import numpy as np
from networks import ActorNetwork, CriticNetwork
from noise import OUActionNoise
from buffer import ReplayBuffer

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions,
                 gamma=0.99, max_size=1000000, fc1_dims = 400, fc2_dims=300,
                 batch_size=64): #64 transition for our batch size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUAactionNoise(mu=np.zeros(n_actions))
        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')
        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name ='target_critic')
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval() # to track of layernorm statistics, we will set it as an evaluate mode
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu =  self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach.numpy()[0]
        # Because mu is pytorch tensor, and env is numpy array.

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

