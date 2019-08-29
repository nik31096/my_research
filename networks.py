import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


# ################ ResNet part ###################


class BasicBlock(nn.Module):
    def __init__(self, in_maps, out_maps, downsample=False):
        super(BasicBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.conv1 = nn.Conv2d(in_maps, out_maps, (3, 3), stride=1 if not downsample else 2, padding=1)
        self.conv2 = nn.Conv2d(out_maps, out_maps, (3, 3), stride=1, padding=1)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_maps, out_maps, (1, 1), stride=2),
                nn.BatchNorm2d(out_maps)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d((3, 3), stride=1, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.layer5 = nn.Sequential(
            BasicBlock(512, 1024, downsample=True),
            BasicBlock(1024, 1024),
            BasicBlock(1024, 1024)
        )
        self.flatten = Flatten()
        self.dense1 = nn.Linear(3*3*1024, 1024)
        self.mu1 = nn.Linear(1024, 1)
        self.sigma1 = nn.Linear(1024, 1)
        self.mu2 = nn.Linear(1024, 1)
        self.sigma2 = nn.Linear(1024, 1)
        self.mu3 = nn.Linear(1024, 1)
        self.sigma3 = nn.Linear(1024, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.flatten(out)
        out = F.relu(self.dense1(out))
        mu1 = self.mu1(out)
        sigma1 = self.sigma1(out)
        mu2 = self.mu2(out)
        sigma2 = self.sigma2(out)
        mu3 = self.mu3(out)
        sigma3 = self.sigma3(out)

        return (mu1, sigma1), (mu2, sigma2), (mu3, sigma3)
