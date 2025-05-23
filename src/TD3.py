# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
import random
import torch
import torch.nn.functional as F
from ModularActor import ActorGraphPolicy
from ModularCritic import CriticGraphPolicy
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):

    def __init__(self, args):

        self.args = args
        self.actor = ActorGraphPolicy(args.limb_obs_size, 1, 1, 1,
                                      args.msg_dim, args.h_dim, args.batch_size,
                                      args.max_action, args.max_children,
                                      args.disable_fold, args.td, args.bu, args.memory).to(device)
        self.actor_target = ActorGraphPolicy(args.limb_obs_size, 1, 1, 1,
                                             args.msg_dim, args.h_dim, args.batch_size,
                                             args.max_action, args.max_children,
                                             args.disable_fold, args.td, args.bu, args.memory).to(device)
        self.critic = CriticGraphPolicy(args.limb_obs_size, 1, 1, 1,
                                        args.msg_dim, args.h_dim, args.batch_size,
                                        args.max_children, args.disable_fold,
                                        args.td, args.bu, args.memory).to(device)
        self.critic_target = CriticGraphPolicy(args.limb_obs_size, 1, 1, 1,
                                               args.msg_dim, args.h_dim, args.batch_size,
                                               args.max_children, args.disable_fold,
                                               args.td, args.bu, args.memory).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

    def change_morphology(self, graph):
        self.actor.change_morphology(graph)
        self.actor_target.change_morphology(graph)
        self.critic.change_morphology(graph)
        self.critic_target.change_morphology(graph)

    def select_action(self, state, action_0=None, r_0=None, h=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            if self.args.memory:
                # Ensure additional inputs are provided and convert to tensors
                if action_0 is None or r_0 is None or h is None:
                    raise ValueError("action_0, r_0, and h must be provided when args.memory is True")
                action_0 = torch.FloatTensor(action_0.reshape(1, -1)).to(device)
                r_0 = torch.FloatTensor(r_0.reshape(1, -1)).to(device)
                h = torch.FloatTensor(h.reshape(1, -1)).to(device)

                # Call actor with memory-related inputs
                action, h = self.actor(state, action_0, r_0, h, 'inference')
                return action.cpu().numpy().flatten(), h.cpu().numpy().flatten()
            else:
                # Call actor with only state
                action = self.actor(state, 'inference')
                return action.cpu().numpy().flatten()

    def train_single(self, replay_buffer, iterations, batch_size=100, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            # select action according to policy and add clipped noise
            with torch.no_grad():
                noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = self.actor_target(next_state) + noise
                next_action = next_action.clamp(-self.args.max_action, self.args.max_action)

                # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (done * discount * target_Q)

            # get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # delayed policy updates
            if it % policy_freq == 0:

                # compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # update the frozen target models
                for param, target_param in zip(self.critic.parameters(),
                                               self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train_single_with_memory(self, replay_buffer, iterations, discount=0.99,
                tau=0.005, policy_noise=0.2, noise_clip=0.5):
        # Sample replay buffer.
        x, y, u_0, u, r_0, r, h, d = replay_buffer.sample()
        # Hint: x.shape should be (batch, step_num, dim).
        for it in range(iterations):
            state = torch.FloatTensor(x[:, it, :]).to(device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            next_state = torch.FloatTensor(y[:, it, :]).to(device)
            if len(next_state.shape) == 1:
                next_state = next_state.unsqueeze(0)
            action_0 = torch.FloatTensor(u_0[:, it, :]).to(device)
            if len(action_0.shape) == 1:
                action_0 = action_0.unsqueeze(0)
            action = torch.FloatTensor(u[:, it, :]).to(device)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            reward_0 = torch.FloatTensor(r_0[:, it, :]).to(device)
            if len(reward_0.shape) == 1:
                reward_0 = reward_0.unsqueeze(0)
            reward = torch.FloatTensor(r[:, it, :]).to(device)
            if len(reward.shape) == 1:
                reward = reward.unsqueeze(0)
            hidden = torch.FloatTensor(h[:, it, :]).to(device)
            if len(hidden.shape) == 1:
                hidden = hidden.unsqueeze(0)
            done = torch.FloatTensor(1 - d[:, it, :]).to(device)
            if len(done.shape) == 1:
                done = done.unsqueeze(0)
            # Select action according to policy and add clipped noise.
            with torch.no_grad():
                noise = action.data.normal_(0, policy_noise).to(device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action, next_h = self.actor_target(next_state, action_0, reward_0, hidden)
                next_action = next_action + noise
                next_action = next_action.clamp(-self.args.max_action, self.args.max_action)
                # Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
                target_Q1, target_Q2 = self.critic_target(next_state, next_action, action, reward, next_h)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward[:,0:1] + (done * discount * target_Q)

            # get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action, action_0, reward_0, hidden)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            action_t, _ = self.actor(state, action_0, reward_0, hidden)
            if len(action_t.shape) == 1:
                action_t = action_t.unsqueeze(0)
            # compute actor loss
            actor_loss = -self.critic.Q1(state, action_t, action_0, reward_0, hidden).mean()

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

    def train(self, replay_buffer_list, iterations_list, batch_size=100, discount=0.99,
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, graphs=None, envs_train_names=None):
        random.shuffle(envs_train_names)
        for env_name in envs_train_names:
            replay_buffer = replay_buffer_list[env_name]
            self.change_morphology(graphs[env_name])
            if self.args.memory:
                train_single_start_t = time.time()

                self.train_single_with_memory(replay_buffer, iterations_list, discount=discount,
                tau=tau, policy_noise=policy_noise, noise_clip=noise_clip)

                train_single_end_t = time.time()
                train_single_time = train_single_end_t - train_single_start_t
                print(f"train_single time: {train_single_time:.6f} sec")
            else:
                self.train_single(replay_buffer, iterations_list, batch_size=1, discount=discount,
                    tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, policy_freq=policy_freq)

        # Update gradient.
        if self.args.memory:
            self.critic_optimizer.step()
            self.actor_optimizer.step()
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, fname):
        torch.save(self.actor.state_dict(), '%s_actor.pth' % fname)
        torch.save(self.critic.state_dict(), '%s_critic.pth' % fname)

    def load(self, fname):
        self.actor.load_state_dict(torch.load('%s_actor.pth' % fname))
        self.critic.load_state_dict(torch.load('%s_critic.pth' % fname))
