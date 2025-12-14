import torch
import torch.nn as nn
# from torch.ao.quantization import QuantStub, DeQuantStub
from collections import OrderedDict
import math
import torch.quantization as tq


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

        # MSR initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        out1 = self.bn2(self.conv2(x))
        out2 = self.bn3(self.conv3(x))
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out

class SqueezeNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNetCIFAR10, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 32 -> 16

        self.fire2 = Fire(96, 16, 64)
        self.fire3 = Fire(128, 16, 64)
        self.fire4 = Fire(128, 32, 128)
        self.dropout4 = nn.Dropout(p=0.2)
        self.maxpool2 = nn.MaxPool2d(2, 2)  # 16 -> 8

        self.fire5 = Fire(256, 32, 128)
        self.fire6 = Fire(256, 48, 192)
        self.fire7 = Fire(384, 48, 192)
        self.fire8 = Fire(384, 64, 256)
        self.dropout8 = nn.Dropout(p=0.2)
        self.maxpool3 = nn.MaxPool2d(2, 2)  # 8 -> 4

        self.fire9 = Fire(512, 64, 256)
        self.conv10 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.dropout_final = nn.Dropout(p=0.5)
        self.avg_pool = nn.AvgPool2d(4)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.dropout4(x)
        x = self.maxpool2(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.dropout8(x)
        x = self.maxpool3(x)

        x = self.fire9(x)
        x = self.conv10(x)
        x = self.dropout_final(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return x
    
    def load_model(self, path='squeezenet_cifar10.pth',device='cpu'):
        state_dict = torch.load(path,map_location=device)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v

        self.load_state_dict(new_state_dict)
        self.to(device)
        self.eval()

        print(f"Model loaded from {path}")
        # print(self)

    def save_model(self, path='squeezenet_cifar10.pth'):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")


class SqueezeNetCIFAR10_QAT(SqueezeNetCIFAR10):
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        # add QAT stubs
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        # quantize input
        x = self.quant(x)

        # original forward
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.dropout4(x)
        x = self.maxpool2(x)

        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.dropout8(x)
        x = self.maxpool3(x)

        x = self.fire9(x)
        x = self.conv10(x)
        x = self.dropout_final(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        # dequantize output
        x = self.dequant(x)
        return x