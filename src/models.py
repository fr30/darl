import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, encoder, num_actions, num_features=1024):
        super().__init__()
        self.encoder = encoder
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(encoder.out_features, num_features, bias=True)),
            nn.ReLU(),
            layer_init(nn.Linear(num_features, num_features, bias=True)),
            nn.ReLU(),
            layer_init(nn.Linear(num_features, num_actions, bias=True)),
        )

    def forward(self, x):
        x = self.encoder(x).detach()
        x = F.relu(x)
        logits = self.trunk(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1) # TODO: Change to regular softmax?

        return action, log_prob, action_probs


class Critic(nn.Module):
    def __init__(self, encoder, num_actions):
        super().__init__()
        self.encoder = encoder
        self.qf1 = QFunction(encoder.out_features, num_actions)
        self.qf2 = QFunction(encoder.out_features, num_actions)

    def forward(self, x):
        x = self.encoder(x)
        q1 = self.qf1(x)
        q2 = self.qf2(x)

        return q1, q2


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class QFunction(nn.Module):
    def __init__(self, in_features, num_actions, num_features=1024):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(in_features, num_features, bias=True)),
            nn.ReLU(),
            layer_init(nn.Linear(num_features, num_features, bias=True)),
            nn.ReLU(),
            layer_init(nn.Linear(num_features, num_actions, bias=True))
        )

    def forward(self, x):
        return self.trunk(x)


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, num_filters=32, out_features=50):
        super().__init__()
        self.out_features = out_features
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[2], num_filters, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            z = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1]).float()
            conv_output_dim = self.conv(z).shape[1]

        self.linear = nn.Linear(conv_output_dim, out_features, bias=True)
        self.lnorm = nn.LayerNorm(out_features, eps=1e-05, elementwise_affine=True)


    def forward(self, x):
        x = x / 255.0
        x = x.transpose(1, 3).transpose(2, 3) # TODO: Dirt hack, figure out how to put it outside of encoder
        x = self.conv(x)
        x = torch.relu
        x = self.linear(x)
        x = self.lnorm(x)

        return x
    