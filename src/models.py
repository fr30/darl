import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[2], 16, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=2, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            z = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            output_dim = self.conv(z).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 128))
        self.fc_q = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3) # Should be moved out of definition of the nn
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[2], 16, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=2, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=2, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            z = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            output_dim = self.conv(z).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 128))
        self.fc_logits = layer_init(nn.Linear(128, envs.single_action_space.n))

    def forward(self, x):
        x = x.transpose(1, 3).transpose(2, 3)
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs