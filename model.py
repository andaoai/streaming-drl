import os, pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F
from torch.distributions import Categorical
from optim import ObGD as Optimizer
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import torch.nn.functional as F
from normalization_wrappers import NormalizeObservation, ScaleReward
from sparse_init import sparse_init

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

class LayerNormalization(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())
    def extra_repr(self) -> str:
        return "Layer Normalization"

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = LayerNormalization()
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = LayerNormalization()
        
        # Shortcut connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        
        out += identity
        out = self.relu(out)
        return out


class ValueNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(ValueNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(8, 32, kernel_size=8, stride=5)
        self.norm1 = nn.LayerNorm([32, 16, 16])  # Adjust shape based on your input size
        self.relu = nn.LeakyReLU()
        self.res_block1 = ResidualBlock(32, 64, stride=3)
        self.res_block2 = ResidualBlock(64, 64, stride=2)
        self.flatten = nn.Flatten(start_dim=0)
        self.fc1 = nn.Linear(576, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        value = self.fc2(out)
        return value

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, n_actions):
        super(PolicyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(8, 32, kernel_size=8, stride=5)
        self.norm1 = nn.LayerNorm([32, 16, 16])  # Adjust shape based on your input size
        self.relu = nn.LeakyReLU()
        self.res_block1 = ResidualBlock(32, 64, stride=3)
        self.res_block2 = ResidualBlock(64, 64, stride=2)
        self.flatten = nn.Flatten(start_dim=0)
        self.fc1 = nn.Linear(576, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        policy = self.fc2(out)
        return policy