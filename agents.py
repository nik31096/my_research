import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import ResNet

import numpy as np

import os


class ReinforceAgent:
    def __init__(self, gamma, entropy_coef):
        self.action_bounds = [
            [-1., 1.],
            [1., 0.],
            [1., 0.]
        ]
        self.model = ResNet()  # will return a 1024-d vector of the state
        self.gamma = gamma
        self.opt = torch.optim.Adam(lr=0.001, params=self.model.parameters())
        self.entropy_coef = entropy_coef

    def act(self, state):
        state = np.ascontiguousarray(state.reshape(3, 96, 96))[np.newaxis, :]
        state = torch.FloatTensor(state)
        (mu1, sigma1), (mu2, sigma2), (mu3, sigma3) = self.model(state)
        self.dist_for_a1 = torch.distributions.normal.Normal(mu1, sigma1)
        self.dist_for_a2 = torch.distributions.normal.Normal(mu2, sigma2)
        self.dist_for_a3 = torch.distributions.normal.Normal(mu3, sigma3)

        a1 = torch.clamp(self.dist_for_a1.sample(), self.action_bounds[0][0], self.action_bounds[0][1])
        a2 = torch.clamp(self.dist_for_a2.sample(), self.action_bounds[1][0], self.action_bounds[1][1])
        a3 = torch.clamp(self.dist_for_a3.sample(), self.action_bounds[2][0], self.action_bounds[2][1])

        return torch.stack([a1, a2, a3]).squeeze()

    def get_log_probs(self, action):
        return torch.stack([self.dist_for_a1.log_prob(action[0]),
                            self.dist_for_a2.log_prob(action[1]),
                            self.dist_for_a3.log_prob(action[2])
                            ]).squeeze()

    def update(self, log_actions_probs, rewards):
        r = self.get_cumulative_rewards(rewards)
        cumulative_rewards = torch.FloatTensor(r)
        J = torch.mean(log_actions_probs * torch.sum(cumulative_rewards))
        entropy = torch.mean(torch.stack([self.dist_for_a1.entropy(),
                                          self.dist_for_a1.entropy(),
                                          self.dist_for_a1.entropy()]))
        loss = -J - self.entropy_coef * entropy
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()

        return loss.data.numpy(), entropy.data.numpy()

    def load_pretrained_model(self, load_from: str):
        """
        Load pretrained checkpoint from file.
        :param load_from: path to directory (without '/' in the end), that contains obs model and act model
        """
        print(f"\n[INFO] Loading model from {load_from}\n")
        try:

            obs_state_dict = torch.load(load_from + '/model.pt')
            self.model.load_state_dict(obs_state_dict)
        except:
            print("[INFO] Failed to load checkpoint...")

    def save_model(self, save_path: str):
        """
        Save checkpoint to file.
        :param save_path: path to directory (without '/' in the end), where to save models
        """
        print(f"\n[INFO] Saving model to {os.path.realpath(save_path) + '/'}")
        try:
            torch.save(self.model.state_dict(), save_path + f'/model.pt')
        except:
            print("[INFO] Failed to save checkpoint...")

    def get_cumulative_rewards(self, rewards):
        def G_t(reward_arr, gamma):
            return sum([gamma ** index * r for index, r in enumerate(reward_arr)])

        G = [G_t(rewards[index:], self.gamma) for index, r in enumerate(rewards)]

        return np.array(G)
