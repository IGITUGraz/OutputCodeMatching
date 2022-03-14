import math
from models.quantization import *

## This script is composed from the following public repositories:
## https://github.com/jiawangbai/TA-LBF
## https://github.com/adnansirajrakin/T-BFA


__all__ = ['CifarResNet', 'ResNet', 'resnet20_quan', 'resnet50_quan', 'resnet20_quan_mid', 'resnet50_quan_mid']


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_output=10, n_bits=8, output_act='linear'):
        super(CifarResNet, self).__init__()
        self.in_planes = 16
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = quan_Linear(64, num_output, n_bits=self.n_bits)
        self.output_act = nn.Tanh() if output_act == 'tanh' else None

        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=self.n_bits))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.output_act(out) if self.output_act is not None else out
        return out


class CifarResNet_mid(nn.Module):
    def __init__(self, block, num_blocks, num_output=10, n_bits=8):
        super(CifarResNet_mid, self).__init__()
        self.in_planes = 16
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = quan_Linear(64, num_output, n_bits=self.n_bits)
        self.mid_dim = 64

        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=self.n_bits))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, n_bits=8):
        super(BasicBlock, self).__init__()
        self.conv1 = quan_Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = DownsampleA(in_planes, planes, stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, base_width=64, n_bits=8):
        super(Bottleneck, self).__init__()
        groups = 1
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = quan_Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = quan_Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = quan_Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False, n_bits=n_bits)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_output=1000, n_bits=8, output_act='linear'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.base_width = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = quan_Linear(512 * block.expansion, num_output, n_bits=self.n_bits)
        self.output_act = nn.Tanh() if output_act == 'tanh' else None

        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        ds = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            ds = nn.Sequential(quan_Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride,
                                           bias=False, n_bits=self.n_bits), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample=ds, base_width=self.base_width, n_bits=self.n_bits))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, downsample=None, base_width=self.base_width, n_bits=self.n_bits))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        out = self.output_act(out) if self.output_act is not None else out
        return out


class ResNet_mid(nn.Module):
    def __init__(self, block, num_blocks, num_output=1000, n_bits=8):
        super(ResNet_mid, self).__init__()
        self.in_planes = 64
        self.base_width = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = quan_Linear(512 * block.expansion, num_output, n_bits=self.n_bits)
        self.mid_dim = 512 * block.expansion

        # Initialize weights
        for m in self.modules():
            if isinstance(m, quan_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        ds = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            ds = nn.Sequential(quan_Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride,
                                           bias=False, n_bits=self.n_bits), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample=ds, base_width=self.base_width, n_bits=self.n_bits))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, downsample=None, base_width=self.base_width, n_bits=self.n_bits))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet20_quan(num_output=10, n_bits=8, output_act='linear'):
    model = CifarResNet(BasicBlock, [3, 3, 3], num_output, n_bits, output_act)
    return model


def resnet20_quan_mid(num_output=10, n_bits=8):
    model = CifarResNet_mid(BasicBlock, [3, 3, 3], num_output, n_bits)
    return model


def resnet50_quan(num_output=1000, n_bits=8, output_act='linear'):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_output, n_bits, output_act)
    return model


def resnet50_quan_mid(num_output=1000, n_bits=8):
    model = ResNet_mid(Bottleneck, [3, 4, 6, 3], num_output, n_bits)
    return model
