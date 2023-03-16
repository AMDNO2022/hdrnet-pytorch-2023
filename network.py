import torch
import torch.nn as nn
import torch.nn.functional as F


class FNet(nn.Module):

    def __init__(self, channel, local_feature_size):
        super(FNet, self).__init__()
        self.channel_size = channel

        self.conv_local = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv_global = nn.Conv2d(64, 64, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channel))

    def forward(self, local_features, global_features): 

        local_features = self.conv_local(local_features)
        global_features = self.conv_global(global_features)
        for channel in range(self.channel_size):
            local_features[channel] = local_features[channel] + global_features[channel] + self.bias[channel]
        return local_features

        
class LowNet(nn.Module):

    def __init__(self, channel):
        super(LowNet, self).__init__()
        self.s1_conv = nn.Conv2d(in_channels=channel, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.s2_conv = nn.Conv2d(8, 16, 3, 2, 1)
        self.s3_conv = nn.Conv2d(16, 32, 3, 2, 1)
        self.s4_conv = nn.Conv2d(32, 64, 3, 2, 1)

        self.l1_conv = nn.Conv2d(64, 64, 3, 1, 1)
        self.l2_conv = nn.Conv2d(64, 64, 3, 1, 1)

        self.g1_conv = nn.Conv2d(64, 64, 3, 2, 1)
        self.g2_conv = nn.Conv2d(64, 64, 3, 2, 1)

        self.g3_fc = nn.Linear(64 * 4 * 4, 256)
        self.g4_fc = nn.Linear(256, 128)
        self.g5_fc = nn.Linear(128, 64)
        self.f_net = FNet(64, 16)
        self.A_fc = nn.Linear(64 * 16 * 16, 64 * 64 * 6)

    def forward(self, x):
        s1 = F.relu(self.s1_conv(x))
        s2 = F.relu(self.s2_conv(s1))
        s3 = F.relu(self.s3_conv(s2))
        s4 = F.relu(self.s4_conv(s3))

        l1 = F.relu(self.l1_conv(s4))
        l2 = F.relu(self.l2_conv(l1))

        g1 = F.relu(self.g1_conv(s4))
        g2 = F.relu(self.g2_conv(g1))
        g3 = F.relu(self.g3_fc(g2.view(1, 64*4*4)))
        g4 = F.relu(self.g4_fc(g3))
        g5 = F.relu(self.g5_fc(g4)).view(64, 1, 1)

        f = F.relu(self.f_net(l2, g5))
        A = F.relu(self.A_fc(f.view(1, 64*16*16)))

        return A.view(1, 6, 64, 64)

class PixelWiseNet(nn.Module):

    def __init__(self, channel):
        super(PixelWiseNet, self).__init__()
        self.conv = nn.Conv2d(channel, channel, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x

class FullNet(nn.Module):

    def __init__(self, channel):
        super(FullNet, self).__init__()
        self.channel_size = channel
        self.p_net = PixelWiseNet(channel)
        self.l_net = LowNet(channel)
        self.unconv1 = nn.ConvTranspose2d(channel * 2, channel * 3, 4, 4, 0)
        self.unconv2 = nn.ConvTranspose2d(channel * 3, channel * 4, 4, 4, 0)
        self.conv1 = nn.Conv2d(channel * 5, channel * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(channel * 3, channel, 1, 1, 0)

    def forward(self, full_img, low_img):
        # input 1024 * 1024
        g = self.p_net(full_img).view(self.channel_size, 1024 * 1024)
        # input 256 * 256
        A = self.l_net(low_img)

        A1 = F.relu(self.unconv1(A))
        A2 = F.relu(self.unconv2(A1)).view(self.channel_size * 4, 1024 * 1024)
        _A = F.relu(self.conv1(torch.cat((g, A2), 0).view(self.channel_size * 5, 1024, 1024)))
        _A1 = torch.cat((_A.view(self.channel_size * 2, 1024 * 1024), full_img.view(self.channel_size, 1024 * 1024)), 0)

        O = self.conv2(_A1.view(self.channel_size * 3, 1024, 1024))

        return O
