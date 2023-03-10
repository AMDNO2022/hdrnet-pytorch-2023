import torch
import torch.nn as nn
import torch.nn.functional as F


class FNet(nn.Module):

    def __init__(self, channel, local_feature_size):
        super(FNet, self).__init__()
        self.channel_size = channel

        self.weight_local = nn.Parameter(torch.Tensor(channel, local_feature_size))
        self.weight_global = nn.Parameter(torch.Tensor(channel))
        self.bias = nn.Parameter(torch.zeros(channel))

        self.weight_local.data.uniform_(-0.1, 0.1)
        self.weight_global.data.uniform_(-0.1, 0.1)

    def forward(self, local_features, global_features, device): 
        result = torch.empty(local_features.shape).to(device)
        for channel in range(self.channel_size):
            result[channel] = self.weight_local[channel] * local_features[channel] + self.weight_global[channel] * global_features[0, channel] + self.bias[channel]
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
        self.A_fc = nn.Linear(64 * 16 * 16, 16 * 16 * 96)

    def forward(self, x, device):
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
        g5 = F.relu(self.g5_fc(g4))

        f = F.relu(self.f_net(l2, g5, device))
        A = F.relu(self.A_fc(f.view(1, 64*16*16)))

        return A.view(16, 16, 8, 3, 4)

class PixelWiseNet(nn.Module):

    def __init__(self, channel):
        super(PixelWiseNet, self).__init__()
        self.channel_size = channel

        if channel == 3:
            self.M = nn.Parameter(torch.Tensor([[1.463, -0.126, 0.201], [0.727, 1.046, 0.521], [0.269, -0.07, 1.368]]))
        else:
            self.M = nn.Parameter(torch.Tensor(channel, channel))
            self.M.data.uniform_(-1, 1)
        self.p_a = nn.Parameter(torch.Tensor(channel, 16))
        self.p_t = nn.Parameter(torch.Tensor(channel, 16))
        self.bias_c = nn.Parameter(torch.zeros(channel))
        self.bias = nn.Parameter(torch.zeros(1))

        self.p_a.data.uniform_(-0.1, 0.1)
        self.p_t.data.uniform_(-0.1, 0.1)

    def forward(self, x, device):
        result = torch.zeros(x.shape[1],x.shape[2]).to(device)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                for channel in range(self.channel_size):
                    
                    def p_c_x(_x, _channel):
                        _result = 0
                        for _i in range(16):
                            _result += self.p_a[channel, _i] * max(_x - self.p_t[channel, _i], 0)
                        return _result

                    result[i, j] += p_c_x((self.M[channel] * x[channel, i, j]).sum() + self.bias_c[channel], channel)

                result[i, j] += self.bias[0]

        return result  

class FullNet(nn.Module):

    def __init__(self, channel):
        super(FullNet, self).__init__()
        self.channel_size = channel
        self.p_net = PixelWiseNet(channel)
        self.l_net = LowNet(channel)

    def forward(self, full_img, low_img, device):
        # input 1024 * 1024
        g = self.p_net(full_img, device)
        g_min = g.min()
        g = (g - g_min) / (g.max() - g_min)
        # input 256 * 256
        A = self.l_net(low_img, device)

        _A = self.grid_slice(g, A, device)
        O = self.apply_coefficients(_A, full_img, device)

        return O
    
    def grid_slice(self, g, A, device):
        # drop edge:
        #           1024 / 16 = 64
        #           (1024 - 64) * (1024 - 64) = 960 * 960
        result = torch.zeros([960, 960, 3, 4]).to(device)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                # 1024 / 16 = 64
                x, y, z = i / 64, j / 64, g[i, j] * 6
                x0, y0, z0 = int(x), int(y), int(z)
                x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
                xd, yd, zd = x - x0, y - y0, z - z0
                
                p1 = A[x0, y0, z0] * xd + A[x1, y0, z0] * (1 - xd)
                p2 = A[x0, y1, z0] * xd + A[x1, y1, z0] * (1 - xd)
                p3 = A[x0, y0, z1] * xd + A[x1, y0, z1] * (1 - xd)
                p4 = A[x0, y1, z1] * xd + A[x1, y1, z1] * (1 - xd)

                q1 = p1 * yd + p2 * (1 - yd)
                q2 = p3 * yd + p4 * (1 - yd)

                result[i, j] = q1 * zd + q2 * (1 - yd)
        return result

    def apply_coefficients(self, _A, x, device):
        result = torch.zeros(self.channel_size, _A.shape[0], _A.shape[1]).to(device)
        for channel in range(self.channel_size):
            for i in range(result.shape[1]):
                for j in range(result.shape[2]):
                    for k in range(self.channel_size):
                        result[channel, i, j] += _A[i, j, channel, k] * x[channel, i, j]
                result[channel, i, j] += _A[i, j, channel, 3]
        return result

