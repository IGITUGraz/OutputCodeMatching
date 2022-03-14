import math
from models.quantization import *

__all__ = ['WideResNet', 'WideResNet_mid', 'wrn28_4_quan', 'wrn28_8_quan', 
           'wrn28_4_quan_mid', 'wrn28_8_quan_mid']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, n_bits=8):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = quan_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = quan_Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False, n_bits=n_bits)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and quan_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                  padding=0, bias=False, n_bits=n_bits) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, n_bits=8):
        super(NetworkBlock, self).__init__()
        self.n_bits = n_bits
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, self.n_bits))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_output=100, n_bits=8, output_act='linear'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False, n_bits=n_bits)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, n_bits)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, n_bits)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, n_bits)
        self.nChannels = nChannels[3]
        self.bn1 = nn.BatchNorm2d(self.nChannels)
        self.relu = nn.ReLU(inplace=True)
        self.linear = quan_Linear(self.nChannels, num_output, n_bits=n_bits)
        self.output_act = nn.Tanh() if output_act == 'tanh' else None

        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, quan_Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.linear(out)
        out = self.output_act(out) if self.output_act is not None else out
        return out


class WideResNet_mid(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_output=100, n_bits=8):
        super(WideResNet_mid, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False, n_bits=n_bits)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, n_bits)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, n_bits)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, n_bits)
        self.nChannels = nChannels[3]
        self.bn1 = nn.BatchNorm2d(self.nChannels)
        self.relu = nn.ReLU(inplace=True)
        self.linear = quan_Linear(self.nChannels, num_output, n_bits=n_bits)
        self.mid_dim = self.nChannels

        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, quan_Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out


def wrn28_4_quan(num_output=100, n_bits=8, output_act='linear'):
    return WideResNet(depth=28, widen_factor=4, num_output=num_output, n_bits=n_bits, output_act=output_act)


def wrn28_8_quan(num_output=100, n_bits=8, output_act='linear'):
    return WideResNet(depth=28, widen_factor=8, num_output=num_output, n_bits=n_bits, output_act=output_act)


def wrn28_4_quan_mid(num_output=100, n_bits=8):
    return WideResNet_mid(depth=28, widen_factor=4, num_output=num_output, n_bits=n_bits)


def wrn28_8_quan_mid(num_output=100, n_bits=8):
    return WideResNet_mid(depth=28, widen_factor=8, num_output=num_output, n_bits=n_bits)
