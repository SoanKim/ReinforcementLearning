# Critic Network

# 2 hidden layers; 400 * 300; ReLU activation
# Adam optimizer 1 * 10^-3; L2 weight decay 0.01
    # Output layer random weights[-3*10^-3, 3*10^-3]
    # Other layers random weights[-1/sqrt(f), 1/sqrt(f)]
# Batch normalization prior to action input(2nd HL)
# Save and load checkpoints

######## My code #########
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.functional as F
#
# class CriticNetwork(nn.Module):
#     def __init__(self, state, action, reward, state_):
#         self.index = 0
#         self.state = state
#         self.state = action
#         self.reward = reward
#         self.state_ = state_
#
#         super(CriticNetwork, self).__init__()
#         hidden1 = nn.Linear(400, 300)
#         hidden2 = nn.Linear(hidden1, function='ReLu')
#         output_layer = np.random.uniform(low=-3*10^-3, high=-3*10^-3, size=(hidden2,))
#
#     def BatchNorm(self, n_batch):
#         reward = []
#         index = 0
#         state = np.zeroes(400, 300)
#         batch = state % n_batch
#
#         self.reward[index]
#
#         index += 1
#         for i in len(batch):
#             hidden2 = nn.BatchNorm2d(axis=1, batch[i])
#
#         return hidden2

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = 'tmp/ddpg'):
    #beta = lr, fc1=inputdim, fc2 = fully connected layer, chkpt_dir = model check pointing
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ddpg')

        # Define the network
        self.fc1 = nn.Linear(*self.inpt_dims, self.fc1_dims) # First take input dim and connect it to fc2 dim
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # Important to normalize input tensors.
        # Here, layer normalization will be used (elem-wise norm) which is independent of batch size.
        # When you copy parameters, batch norm doesn't keep track of running mean and running var.
        # You gotta use flag with long state dict equls false
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1) # critic's value
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0]) # fan (what is fan?)
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.actions_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.vn1(state_value)
        state_value = F.relu(state_value)
        # Activate before normalization, because it not, negative values can be lopped off.
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        # concatenating is not good
        # ∵ Q table should have # of col & rows.
        # cols = each action for state (baseline val 2 the each state). The val of the actions tells u what u're gonna gain through that action.
        # The shape will be n_actions (state values) +1  in shape. Wonky in dimensionality wise. Displacement of velocity.
        # = What u're doing in matched with incorrect # of dim. Incorrect.
        # Relu on both state and action function.
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dic(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))