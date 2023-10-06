import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import RandomCrop
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, encoder, num_actions, num_features=1024):
        super().__init__()
        self.encoder = encoder
        self.trunk = nn.Sequential(
            nn.Linear(encoder.out_features, num_features, bias=True),
            nn.ReLU(),
            nn.Linear(num_features, num_actions, bias=True),
        )

    def forward(self, x):
        x = self.encoder(x).detach()
        x = F.relu(x)
        logits = self.trunk(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class DoubleQNetwork(nn.Module):
    def __init__(self, encoder, num_actions, num_features):
        super().__init__()
        self.encoder = encoder
        self.qf1 = QFunction(encoder.out_features, num_actions, num_features)
        self.qf2 = QFunction(encoder.out_features, num_actions, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        q1 = self.qf1(x)
        q2 = self.qf2(x)
        return q1, q2


class QNetwork(nn.Module):
    def __init__(self, encoder, num_features, num_actions):
        super().__init__()
        self.encoder = encoder
        self.qf = QFunction(encoder.out_features, num_actions, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        x = self.qf(x)
        return x


class QFunction(nn.Module):
    def __init__(self, in_features, num_actions, num_features=1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_features, num_features, bias=True),
            nn.ReLU(),
            nn.Linear(num_features, num_features, bias=True),
            nn.ReLU(),
            nn.Linear(num_features, num_actions),
        )

    def forward(self, x):
        return self.trunk(x)


class PixelEncoder(nn.Module):
    def __init__(self, channels, img_size, crop=False, num_filters=32, out_features=50):
        super().__init__()
        self.out_features = out_features
        self.should_crop = crop
        self.crop = RandomCrop(img_size) if crop else None
        self.conv = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, stride=1),
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
