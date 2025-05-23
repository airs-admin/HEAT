from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import MLPBase
from torchfold import Fold
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from config import *

        
class ActorVanilla(nn.Module):
    """
    A vanilla actor module (without memory) that outputs a node's action given only its observation.
    No message passing between nodes.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorVanilla, self).__init__()
        self.max_action = max_action
        self.base = MLPBase(state_dim, action_dim)

    def forward(self, x):
        x = self.max_action * torch.tanh(self.base(x))
        return x


class ActorUp_MemAug(nn.Module):
    """
    A bottom-up module (with memory) used in bothway message passing that only passes message to its parent.
    """
    def __init__(self, state_dim, action_dim_0, reward_dim_0, msg_dim, h_dim, max_children):        
        super(ActorUp_MemAug, self).__init__()
        self.obs_encoder = nn.Linear(state_dim, 64)
        self.act_encoder = nn.Linear(action_dim_0, 64)
        self.rew_encoder = nn.Linear(reward_dim_0, 64)
        self.msg_encoder= nn.Linear(msg_dim * max_children, 64)
        self.add_h_encoder = nn.Linear(64 + 64 + 64 + 64 + h_dim, 64)
        self.msg_decoder = nn.Linear(64, msg_dim)
        self.h_decoder = nn.Linear(64, h_dim)

    def forward(self, x, u, r, h, *m):
        m = torch.cat(m, dim=-1)
        x = F.normalize(self.obs_encoder(x), dim=-1)
        u = F.normalize(self.act_encoder(u), dim=-1)
        r = F.normalize(self.rew_encoder(r), dim=-1)
        m = F.normalize(self.msg_encoder(m), dim=-1)
        combined = torch.cat([x, m, u, r, h], dim=-1)
        combined = torch.tanh(self.add_h_encoder(torch.tanh(combined)))
        msg_up = F.normalize(self.msg_decoder(combined), dim=-1)
        new_hidden = torch.tanh(self.h_decoder(combined))
        return msg_up, new_hidden
    

class ActorUp(nn.Module):
    """
    A bottom-up module (without memory) used in bothway message passing that only passes message to its parent.
    """
    def __init__(self, state_dim, msg_dim, max_children):
        super(ActorUp, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)

    def forward(self, x, *m):
        m = torch.cat(m, dim=-1)
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        xm = self.fc2(xm)
        xm = torch.tanh(xm)
        xm = self.fc3(xm)
        xm = F.normalize(xm, dim=-1)
        msg_up = xm
        return msg_up
    

class ActorUpAction(nn.Module):
    """
    A bottom-up module (without memory) used in bottom-up-only message passing that passes message to its parent and outputs action.
    """
    def __init__(self, state_dim, msg_dim, max_children, action_dim, max_action):
        super(ActorUpAction, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + msg_dim * max_children, 64)
        self.fc3 = nn.Linear(64, msg_dim)
        self.action_base = MLPBase(state_dim + msg_dim * max_children, action_dim)
        self.max_action = max_action

    def forward(self, x, *m):
        m = torch.cat(m, dim=-1)
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        xm = torch.cat([x, m], dim=-1)
        xm = torch.tanh(xm)
        xm = self.fc2(xm)
        xm = torch.tanh(xm)
        xm = self.fc3(xm)
        xm = F.normalize(xm, dim=-1)
        msg_up = xm
        return msg_up, action


class ActorDownAction_MemAug(nn.Module):
    """
    A top-down module (with memory) used in bothway message passing that passes messages to children and outputs action.
    """
    def __init__(self, self_input_dim, action_dim, action_dim_0, reward_dim_0, msg_dim, h_dim, max_action, max_children):
        super(ActorDownAction_MemAug, self).__init__()
        self.max_action = max_action
        self.act_decoder = MLPBase(64 + 64 + 64 + 64 + h_dim, action_dim)
        self.msg_decoder = MLPBase(64 + 64 + 64 + 64 + h_dim, msg_dim * max_children)
        self.h_decoder = MLPBase(64 + 64 + 64 + 64 + h_dim, h_dim)
        self.obs_encoder = nn.Linear(self_input_dim, 64)
        self.msg_encoder = nn.Linear(msg_dim, 64)
        self.act_encoder = nn.Linear(action_dim_0, 64)
        self.rew_encoder = nn.Linear(reward_dim_0, 64)
        
    def forward(self, x, a, r, m, h):
        x = F.normalize(self.obs_encoder(x), dim=-1)
        m = F.normalize(self.msg_encoder(m), dim=-1)
        a = F.normalize(self.act_encoder(a), dim=-1)
        r = F.normalize(self.rew_encoder(r), dim=-1)
        combined = torch.tanh(torch.cat([x, m, a, r, h], dim=-1))
        action = self.max_action * torch.tanh(self.act_decoder(combined))
        msg_down = F.normalize(self.msg_decoder(combined), dim=-1)
        new_hidden = torch.tanh(self.h_decoder(combined))
        return action, msg_down ,new_hidden
    
    
class ActorDownAction(nn.Module):
    """
    A top-down module (without memory) used in bothway message passing that passes messages to children and outputs action.
    """
    # input dim is state dim if only using top down message passing
    # if using bottom up and then top down, it is the node's outgoing message dim
    def __init__(self, self_input_dim, action_dim, msg_dim, max_action, max_children):
        super(ActorDownAction, self).__init__()
        self.max_action = max_action
        self.action_base = MLPBase(self_input_dim + msg_dim, action_dim)
        self.msg_base = MLPBase(self_input_dim + msg_dim, msg_dim * max_children)

    def forward(self, x, m):
        xm = torch.cat((x, m), dim=-1)
        xm = torch.tanh(xm)
        action = self.max_action * torch.tanh(self.action_base(xm))
        msg_down = self.msg_base(xm)
        msg_down = F.normalize(msg_down, dim=-1)
        return action, msg_down


class ActorGraphPolicy(nn.Module):
    """
    A weight-sharing dynamic graph policy that changes its structure based on different morphologies
    and passes messages between nodes.
    """
    def __init__(self, state_dim, action_dim, action_dim_0, reward_dim_0, msg_dim, h_dim, batch_size, max_action, max_children, disable_fold, td, bu, memory):
        super(ActorGraphPolicy, self).__init__()
        self.num_limbs = 1
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.h = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action_0 = [None] * self.num_limbs
        self.input_reward_0 = [None] * self.num_limbs
        self.input_h = [None] * self.num_limbs
        self.max_action = max_action
        self.msg_dim = msg_dim
        self.batch_size = batch_size
        self.max_children = max_children
        self.disable_fold = disable_fold
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_dim_0 = action_dim_0
        self.reward_dim_0 = reward_dim_0
        self.memory = memory
        self.h_dim = h_dim
        self.td = td
        self.bu = bu
        assert self.action_dim == 1
        
        # If memory is enabled, we need to have both bottom-up and top-down message passing
        if self.memory:
            # Bottom-up and top-down
            self.sNet = nn.ModuleList([ActorUp_MemAug(state_dim, action_dim_0, reward_dim_0, msg_dim, h_dim, max_children)] * self.num_limbs).to(device)
            self.actor = nn.ModuleList([ActorDownAction_MemAug(msg_dim, action_dim, action_dim_0, reward_dim_0, msg_dim, h_dim, max_action, max_children)] * self.num_limbs).to(device)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
                    setattr(self, "actor" + str(i).zfill(3), self.actor[i])
        # If memory is disabled, we can choose to have either bottom-up or top-down message passing
        else:
            if self.bu:
                # bottom-up then top-down
                if self.td:
                    self.sNet = nn.ModuleList([ActorUp(state_dim, msg_dim, max_children)] * self.num_limbs).to(device)
                # bottom-up only
                else:
                    self.sNet = nn.ModuleList([ActorUpAction(state_dim, msg_dim, max_children, action_dim, max_action)] * self.num_limbs).to(device)
                if not self.disable_fold:
                    for i in range(self.num_limbs):
                        setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
            # we pass msg_dim as first argument because in both-way message-passing, each node takes in its passed-up message as 'state'
            if self.td:
                # bottom-up then top-down
                if self.bu:
                    self.actor = nn.ModuleList([ActorDownAction(msg_dim, action_dim, msg_dim, max_action, max_children)] * self.num_limbs).to(device)
                # top-down only
                else:
                    self.actor = nn.ModuleList([ActorDownAction(state_dim, action_dim, msg_dim, max_action, max_children)] * self.num_limbs).to(device)
                if not self.disable_fold:
                    for i in range(self.num_limbs):
                        setattr(self, "actor" + str(i).zfill(3), self.actor[i])

            # no message passing
            if not self.bu and not self.td:
                self.actor = nn.ModuleList([ActorVanilla(state_dim, action_dim, max_action)] * self.num_limbs).to(device)
                if not self.disable_fold:
                    for i in range(self.num_limbs):
                        setattr(self, "actor" + str(i).zfill(3), self.actor[i])

        if not self.disable_fold:
            for i in range(self.max_children):
                setattr(self, f'get_{i}', self.addFunction(i))

    def forward(self, state, action_0=None, reward_0=None, h=None, mode='train'):
        """
        Forward pass of the graph policy.
        """
        self.clear_buffer()
        if mode == 'inference':
            original_batch_size = self.batch_size
            self.batch_size = 1
        if not self.disable_fold:
            self.fold = Fold()
            self.fold.cuda()
            self.zeroFold_td = self.fold.add("zero_func_td")
            self.zeroFold_bu = self.fold.add("zero_func_bu")
            self.a = []
        assert state.shape[1] == self.state_dim * self.num_limbs, 'state.shape[1] expects {} but got {} with num_limbs being {} and state_dim being {}'.format(self.state_dim * self.num_limbs, state.shape[1], self.num_limbs, self.state_dim)

        for i in range(self.num_limbs):
            self.input_state[i] = state[:, i * self.state_dim:(i + 1) * self.state_dim]
            if self.memory:             
                self.input_action_0[i] = action_0[:, i].unsqueeze(-1)
                self.input_reward_0[i] = reward_0[:, i].unsqueeze(-1)
                self.input_h[i] = h[:, i * self.h_dim:(i + 1) * self.h_dim]
            if not self.disable_fold:
                self.input_state[i] = torch.unsqueeze(self.input_state[i], 0)

        if self.bu:
            # bottom up transmission by recursion
            for i in range(self.num_limbs):
                self.bottom_up_transmission(i)

        if self.td:
            # top down transmission by recursion
            for i in range(self.num_limbs):
                self.top_down_transmission(i)

        if not self.bu and not self.td:
            for i in range(self.num_limbs):
                if not self.disable_fold:
                    self.action[i] = self.fold.add('actor' + str(0).zfill(3), self.input_state[i])
                else:
                    self.action[i] = self.actor[i](self.input_state[i])

        if not self.disable_fold:
            self.a += self.action
            self.action = self.fold.apply(self, [self.a])[0]
            self.action = torch.transpose(self.action, 0, 1)
            self.fold = None
        else:
            self.action = torch.stack(self.action, dim=-1)
            self.msg_down = torch.stack(self.msg_down, dim=-1)
            if self.memory:
                self.h = torch.stack(self.h, dim=-1)

        if mode == 'inference':
            self.batch_size = original_batch_size

        if self.memory:
            return torch.squeeze(self.action), torch.flatten(self.h, start_dim=1)
        else: 
            return torch.squeeze(self.action)

    def bottom_up_transmission(self, node):

        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_bu
            else:
                return torch.zeros((self.batch_size, self.msg_dim), requires_grad=True).to(device) , torch.zeros((self.batch_size, self.h_dim), requires_grad=True).to(device)

        if self.msg_up[node] is not None:
            return self.msg_up[node]

        state = self.input_state[node]

        children = [i for i, x in enumerate(self.parents) if x == node]
        assert (self.max_children - len(children)) >= 0
        children += [-1] * (self.max_children - len(children))
        msg_in = [None] * self.max_children

        if self.memory:
            for i in range(self.max_children): 
                msg_in[i], h = self.bottom_up_transmission(children[i])
            action_0 = self.input_action_0[node]
            reward_0 = self.input_reward_0[node]
            h = self.input_h[node]
            if not self.disable_fold:
                if self.td:
                    self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, *msg_in)
                else:
                    self.msg_up[node], self.action[node] = self.fold.add('sNet' + str(0).zfill(3), state, *msg_in).split(2)
            else:
                if self.td:
                    self.msg_up[node], self.h[node] = self.sNet[node](state, action_0, reward_0, h, *msg_in)
                else:
                    self.msg_up[node], self.action[node] = self.sNet[node](state, h, *msg_in)
        else:
            for i in range(self.max_children): 
                msg_in[i] = self.bottom_up_transmission(children[i])
            if not self.disable_fold:
                if self.td:
                    self.msg_up[node] = self.fold.add('sNet' + str(0).zfill(3), state, *msg_in)
                else:
                    self.msg_up[node], self.action[node] = self.fold.add('sNet' + str(0).zfill(3), state, *msg_in).split(2)
            else:
                if self.td:
                    self.msg_up[node] = self.sNet[node](state, *msg_in)
                else:
                    self.msg_up[node], self.action[node] = self.sNet[node](state, *msg_in)

        if self.memory:
            return self.msg_up[node], self.h[node]
        else:
            return self.msg_up[node]

    def top_down_transmission(self, node):
        if node < 0:
            if not self.disable_fold:
                return self.zeroFold_td
            else:
                return torch.zeros((self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

        elif self.msg_down[node] is not None:
            return self.msg_down[node]

        # in both-way message-passing, each node takes in its passed-up message as 'state'
        if self.bu:
            state = self.msg_up[node]
        else:
            state = self.input_state[node]
        parent_msg = self.top_down_transmission(self.parents[node])

        # find self children index (first child of parent, second child of parent, etc)
        # by finding the number of previous occurences of parent index in the list
        self_children_idx = self.parents[:node].count(self.parents[node])

        # if the structure is flipped, flip message order at the root
        if self.parents[0] == -2 and node == 1:
            self_children_idx = (self.max_children - 1) - self_children_idx

        if not self.disable_fold:
            msg_in = self.fold.add('get_{}'.format(self_children_idx), parent_msg)
        else:
            msg_in = self.msg_slice(parent_msg, self_children_idx)

        if self.memory:
            action_0 = self.input_action_0[node]
            reward_0 = self.input_reward_0[node]
            h = self.input_h[node]
            if not self.disable_fold:
                self.action[node], self.msg_down[node] = self.fold.add('actor' + str(0).zfill(3), state, msg_in).split(2)
            else:
                self.action[node], self.msg_down[node], self.h[node] = self.actor[node](state, action_0, reward_0, msg_in, h)
        else:
            if not self.disable_fold:
                self.action[node], self.msg_down[node] = self.fold.add('actor' + str(0).zfill(3), state, msg_in).split(2)
            else:
                self.action[node], self.msg_down[node] = self.actor[node](state, msg_in)

        return self.msg_down[node]

    def zero_func_td(self):
        return torch.zeros((1, self.batch_size, self.msg_dim * self.max_children), requires_grad=True).to(device)

    def zero_func_bu(self):
        return torch.zeros((1, self.batch_size, self.msg_dim), requires_grad=True).to(device)

    # an ugly way to define functions in a for loop (for torchfold only)
    def addFunction(self, n):
        def f(x):
            return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[n]
        return f

    def msg_slice(self, x, idx):
        return torch.split(x, x.shape[-1] // self.max_children, dim=-1)[idx]

    def clear_buffer(self):
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        self.input_action_0 = [None] * self.num_limbs
        self.input_reward_0 = [None] * self.num_limbs
        self.input_h = [None] * self.num_limbs
        self.h = [None] * self.num_limbs
        self.zeroFold_td = None
        self.zeroFold_bu = None
        self.fold = None

    def change_morphology(self, parents):
        if not self.disable_fold:
            if self.bu:
                for i in range(1, self.num_limbs):
                    delattr(self, "sNet" + str(i).zfill(3))
            if not (self.bu and not self.td):
                for i in range(1, self.num_limbs):
                    delattr(self, "actor" + str(i).zfill(3))
        self.parents = parents
        self.num_limbs = len(parents)
        self.msg_down = [None] * self.num_limbs
        self.msg_up = [None] * self.num_limbs
        self.action = [None] * self.num_limbs
        self.input_state = [None] * self.num_limbs
        if self.memory:
            self.input_action_0 = [None] * self.num_limbs
            self.input_reward_0 = [None] * self.num_limbs
            self.input_h = [None] * self.num_limbs
        if self.bu:
            self.sNet = nn.ModuleList([self.sNet[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "sNet" + str(i).zfill(3), self.sNet[i])
        if not (self.bu and not self.td):
            self.actor = nn.ModuleList([self.actor[0]] * self.num_limbs)
            if not self.disable_fold:
                for i in range(self.num_limbs):
                    setattr(self, "actor" + str(i).zfill(3), self.actor[i])
