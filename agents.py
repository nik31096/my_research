import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import ResNet

import numpy as np

import os


class ReinforceAgent:
    def __init__(self):
        self.backbone = ResNet()  # will return a 1024-d vector of the state

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

    def act(self, state):
        pass

    def update(self, actions_probs, rewards):
        """
        Updating agent parameters
        :param actions_probs: probabilities of taken actions
        :param rewards: sequence of rewards in rollouts
        :return: loss and entropy
        """
        print("rewards", rewards)
        cumulative_rewards = torch.FloatTensor(self.get_cumulative_rewards(rewards)).to(self.device)
        entropy = -torch.mean(actions_probs * torch.log(actions_probs))
        J = torch.mean(torch.log(actions_probs) * torch.sum(cumulative_rewards))
        self.loss = -J - self.entropy_coef * entropy
        self.loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return self.loss.cpu().data.numpy(), entropy.cpu().data.numpy()