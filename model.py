import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_init import sparse_init
import torchvision.models as models

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
    

# 定义一个恒等映射的模块
class Identity(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(Identity, self).__init__()

    def forward(self, x):
        # 前向传播函数，直接返回输入
        return x

# 定义一个SimCLR模型
class SimCLR(nn.Module):
    def __init__(self, linear_eval=False):
        # 调用父类的构造函数
        super().__init__()
        self.linear_eval = linear_eval  # 是否进行线性评估的标志
        resnet18 = models.resnet18(pretrained=False)  # 加载ResNet-18模型，不使用预训练权重
        resnet18.fc = Identity()  # 将ResNet的全连接层替换为恒等映射
        self.encoder = resnet18  # 将修改后的ResNet作为编码器
        # 定义投影头，由两个线性层和一个ReLU激活函数组成
        self.projection = nn.Sequential(
            nn.Linear(512, 256),  # 第一个线性层，将输入特征从512维映射到256维
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(256, 64)  # 第二个线性层，将输入特征从256维映射到64维
        )

    def forward(self, x):
        # 前向传播函数
        if not self.linear_eval:
            # 如果不是线性评估，拼接输入张量
            x = torch.cat(x, dim=0)  # 在第0维拼接输入的多个样本

        encoding = self.encoder(x)  # 通过编码器获取特征表示
        projection = self.projection(encoding)  # 通过投影头获取投影特征
        return projection  # 返回投影特征