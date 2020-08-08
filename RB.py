# Replay Buffer

# Fixed size, overwrite early memories
    # Sample memories uniformly
    # __init__()
    # store_transition()
    # stroe_transition()
# Should be generic --> no casting to Pytorch tensors
# Recommend using numpy arrays (Compatible to either Keras, Pytorch, etc)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from OUP import QUActionsNoise
# call MinMaxScaler object
min_max_scaler = MinMaxScaler()
# # feed in a numpy array
# X_train_norm = min_max_scaler.fit_transform(X_train.values)
# # wrap it up if you need a dataframe
# df = pd.DataFrame(X_train_norm)

# My code:
# class ReplayBuffer():
#     def __init__(self, state):
#         memory = []
#         state = MinMaxScaler(state.item())
#         action = np.sample(state)
#         memory.append(action)
#
#     def store_transition(self, state):
#         transition = []
#         state = ReplayBuffer(self, state)
#         transition.append(QUActionsNoise(state))

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions): # max size of memory buffer, input shape of the observation from the env. and number of actions for action space
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions)) # actions are continuous numbers.
        # n_action is a misnomer, not really the # of actions but # of components of the action.
        # In case of lunar lander env, it was # of four vector elem that extends to which four engines in the space craft.
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr+=1 # for index

    def sample_buffer(self, batch_size): # Sample the memory buffer uniformly
        # How far are we going to sample? Initialize all the arrays with zeros, which obviously won't teach anything to our agent.
        # We will sample the minimum value depending on the n_actions or batch size, because if we choose the max, it will bring an error.
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones