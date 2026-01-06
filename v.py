import torch
import torch.nn as nn
import yaml

# 定义模型模块
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))

class C2f_DiSpAM(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, use_residual, attention_ratio):
        super(C2f_DiSpAM, self).__init__()
        layers = []
        for _ in range(repeats):
            layers.append(Conv(in_channels, out_channels, 3, 1))
        self.layers = nn.Sequential(*layers)
        self.use_residual = use_residual
        self.attention_ratio = attention_ratio

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class CASAtt(nn.Module):
    def __init__(self):
        super(CASAtt, self).__init__()
        self.att = nn.Sigmoid()

    def forward(self, x):
        return x * self.att(x)

class A2C2f(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, use_residual, attention_ratio):
        super(A2C2f, self).__init__()
        layers = []
        for _ in range(repeats):
            layers.append(Conv(in_channels, out_channels, 3, 1))
        self.layers = nn.Sequential(*layers)
        self.use_residual = use_residual
        self.attention_ratio = attention_ratio

    def forward(self, x):
        if self.use_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MFM(nn.Module):
    def __init__(self, in_channels):
        super(MFM, self).__init__()
        self.conv = Conv(in_channels, in_channels, 1, 1)

    def forward(self, x):
        return self.conv(x)

class Detect(nn.Module):
    def __init__(self, in_channels, nc):
        super(Detect, self).__init__()
        self.conv = nn.Conv2d(in_channels, nc, 1, 1)

    def forward(self, x):
        return self.conv(x)

# 加载 YAML 文件
with open('E:/yolo12/yolov12.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 构建模型
class YOLOv12(nn.Module):
    def __init__(self, cfg):
        super(YOLOv12, self).__init__()
        layers = []
        for layer in cfg['backbone']:
            if layer[2] == 'Conv':
                layers.append(Conv(layer[3][0], layer[3][1], layer[3][2], layer[3][2]))
            elif layer[2] == 'C2f_DiSpAM':
                layers.append(C2f_DiSpAM(layer[3][0], layer[3][1], layer[1], layer[3][2], layer[3][3]))
            elif layer[2] == 'C3k2':
                layers.append(C2f_DiSpAM(layer[3][0], layer[3][1], layer[1], layer[3][2], layer[3][3]))
            elif layer[2] == 'CASAtt':
                layers.append(CASAtt())
            elif layer[2] == 'A2C2f':
                layers.append(A2C2f(layer[3][0], layer[3][1], layer[1], layer[3][2], layer[3][3]))
        self.backbone = nn.Sequential(*layers)

        layers = []
        for layer in cfg['head']:
            if layer[2] == 'nn.Upsample':
                layers.append(nn.Upsample(scale_factor=layer[3][1], mode=layer[3][2]))
            elif layer[2] == 'MFM':
                layers.append(MFM(layer[3][0]))
            elif layer[2] == 'Concat':
                layers.append(nn.Identity())  # Concat is not implemented here
            elif layer[2] == 'Detect':
                layers.append(Detect(layer[3][0], cfg['nc']))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# 实例化模型
model = YOLOv12(cfg)

# 打印模型结构
print(model)