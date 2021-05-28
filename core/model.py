import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
        self.bn = nn.BatchNorm2d(out_channels, eps = 0.001, momentum = 0.1, affine = True)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace = False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class ChannelShuffle(nn.Module):
    def __init__(self):
        super(ChannelShuffle, self).__init__()

    def forward(self, x, groups):
        b, in_c, h, w = x.shape
        return x.view(b, groups, in_c // groups, h, w)\
        .transpose(1, 2)\
        .contiguous()\
        .view(b, -1, h, w)

class DepthwiseInceptionShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels = None, skip_connection = True, downsample = False, **kwargs):
        super(DepthwiseInceptionShuffleBlock, self).__init__()

        if skip_connection:
            assert not out_channels, f'\n\nArgument `out_channels` is redundent when skip connection is provided.\n'
        else:
            assert out_channels, f'\n\nArgument `out_channels` is required when no skip connection is provided.\n'
        
        if downsample:
            stride = 2
            self.downsample = nn.AvgPool2d(2, 2)
        else:
            stride = 1
            self.downsample = None
        
        if skip_connection:
            out_channels = in_channels
        self.skip_connection = skip_connection
        self.grouped_pointwise_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, groups = 4, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.depthwise_3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, groups = out_channels, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.grouped_pointwise_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 1, groups = 8, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
        self.depthwise_5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 5, groups = out_channels, stride = stride, padding = 2, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        self.channel_shuffle = ChannelShuffle()

    def forward(self, x):
        x_3x3 = self.grouped_pointwise_3x3(x)
        x_3x3 = self.channel_shuffle(x_3x3, 4)
        x_3x3 = self.depthwise_3x3(x_3x3) # (-1, 512, 64, 64)
        x_5x5 = self.grouped_pointwise_5x5(x)
        x_5x5 = self.channel_shuffle(x_5x5, 8)
        x_5x5 = self.depthwise_5x5(x_5x5) # (-1, 512, 64, 64)
        
        if self.downsample:
            x = self.downsample(x)
        
        if self.skip_connection:
            return x + x_3x3 + x_5x5
        return x_3x3 + x_5x5

class DISNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv2d_1a = BasicConv2d(3, 32, kernel_size = 3, stride = 2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size = 3, stride = 1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        
        self.maxpool_3a = nn.MaxPool2d(3, stride = 2)
        
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size = 1, stride = 1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size = 3, stride = 1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size = 3, stride = 2)

        self.repeat_1 = nn.Sequential(
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
            Block35(scale = 0.17),
        )

        self.depthwise_inception_shuffle = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            DepthwiseInceptionShuffleBlock(256, 512, skip_connection = False, downsample = True),
            DepthwiseInceptionShuffleBlock(512, 512, skip_connection = False, downsample = True),
            DepthwiseInceptionShuffleBlock(512, 512, skip_connection = False),
        )

        self.feature_embeddings = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(512, 512, (3, 3), groups = 64, bias = False),
            nn.Flatten(),
            nn.ReLU(inplace = True),
        )

        self.logits = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)

        x = self.repeat_1(x)

        x = self.depthwise_inception_shuffle(x)
        x = self.feature_embeddings(x)

        return self.logits(x)