import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import RandomCrop
from torch.distributions.categorical import Categorical
from src.utils import layer_init


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
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x).detach()
        logits = self.trunk(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        # TODO: Change to regular softmax?
        log_prob = F.log_softmax(logits, dim=1)

        return action, log_prob, action_probs


class Critic(nn.Module):
    def __init__(self, encoder, num_actions, num_features):
        super().__init__()
        self.encoder = encoder
        self.qf1 = QFunction(encoder.out_features, num_actions, num_features)
        self.qf2 = QFunction(encoder.out_features, num_actions, num_features)

    def forward(self, x):
        x = self.encoder(x)
        q1 = self.qf1(x)
        q2 = self.qf2(x)

        return q1, q2


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended
# for SAC without stopping actor gradients
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
            layer_init(nn.Linear(num_features, num_actions, bias=True)),
        )

    def forward(self, x):
        return self.trunk(x)


class QNetwork(nn.Module):
    def __init__(self, channels, img_size, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.inference_mode():
            z = torch.zeros(1, channels, img_size, img_size).float()
            conv_output_dim = self.conv(z).shape[1]

        self.trunk = nn.Sequential(
            nn.Linear(conv_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            # F.hardtanh(input, min_val=- 1., max_val=1.
            # nn.Sigmoid()
            # nn.LayerNorm(num_actions, eps=1e-05, elementwise_affine=True),
        )

    def forward(self, x):
        x = self.conv(x / 255.0)
        x = self.trunk(x)
        # x = F.hardtanh(x, 0, 1)
        return x


class PixelEncoder(nn.Module):
    def __init__(self, channels, img_size, crop=False, num_filters=32, out_features=50):
        super().__init__()
        self.out_features = out_features
        self.should_crop = crop
        self.crop = RandomCrop(img_size) if crop else None
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(channels, num_filters, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.inference_mode():
            z = torch.zeros(1, channels, img_size, img_size).float()
            conv_output_dim = self.conv(z).shape[1]

        self.linear = nn.Linear(conv_output_dim, out_features, bias=True)
        self.lnorm = nn.LayerNorm(out_features, eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        x = x / 255.0
        if self.should_crop:
            x = self.crop(x)
        x = self.conv(x)
        x = self.linear(x)
        x = self.lnorm(x)

        return x
