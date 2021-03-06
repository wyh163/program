'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * nn.ReLU6(inplace=True)(x+3) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = nn.ReLU6(inplace=True)(x+3) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = SeModule(expand_size) if semodule==True else None

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = hswish() if nolinear=='hard_swish' else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = hswish() if nolinear=='hard_swish' else nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        if self.se is not None:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV3(nn.Module):
    def __init__(self, model_name, scale, num_classes=1000):
        super(MobileNetV3, self).__init__()
        scale = scale
        large_stride = [1, 2, 2, 2]
        small_stride = [2, 2, 2, 2]
        feas = {}
        feas['large'] = {0.35:336, 0.5:480, 0.75:720, 1.0:960, 1.25:1200}
        feas['small'] = {0.35:200, 0.5:288, 0.75:432, 1.0:576, 1.25:720}
        inplanes = 16

        if model_name == 'large':
            self.stage1 = nn.Sequential(
                Block(3, inplanes, self.make_divisible(scale * 16), 
                    self.make_divisible(scale * 16), 'relu', False, 1),

                Block(3, self.make_divisible(scale * 16), self.make_divisible(scale * 64),
                    self.make_divisible(scale * 24), 'relu', False, 2),

                Block(3, self.make_divisible(scale * 24), self.make_divisible(scale * 72),
                    self.make_divisible(scale * 24), 'relu', False, 1),

                Block(5, self.make_divisible(scale * 24), self.make_divisible(scale * 72),
                    self.make_divisible(scale * 40), 'relu', True, 2),

                Block(5, self.make_divisible(scale * 40), self.make_divisible(scale * 120),
                    self.make_divisible(scale * 40), 'relu', True, 1)
                )

            self.stage2 = nn.Sequential(
                Block(5, self.make_divisible(scale * 40), self.make_divisible(scale * 120),
                    self.make_divisible(scale * 40), 'relu', True, 1),

                Block(3, self.make_divisible(scale * 40), self.make_divisible(scale * 240),
                    self.make_divisible(scale * 80), 'hard_swish', False, 2),

                Block(3, self.make_divisible(scale * 80), self.make_divisible(scale * 200),
                    self.make_divisible(scale * 80), 'hard_swish', False, 1),

                Block(3, self.make_divisible(scale * 80), self.make_divisible(scale * 184),
                    self.make_divisible(scale * 80), 'hard_swish', False, 1),

                Block(3, self.make_divisible(scale * 80), self.make_divisible(scale * 184),
                    self.make_divisible(scale * 80), 'hard_swish', False, 1)
                )

            self.stage3 = nn.Sequential(
                Block(3, self.make_divisible(scale * 80), self.make_divisible(scale * 480),
                    self.make_divisible(scale * 112), 'hard_swish', True, 1),

                Block(3, self.make_divisible(scale * 112), self.make_divisible(scale * 672),
                    self.make_divisible(scale * 112), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 112), self.make_divisible(scale * 672),
                    self.make_divisible(scale * 160), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 160), self.make_divisible(scale * 672),
                    self.make_divisible(scale * 160), 'hard_swish', True, 2),

                Block(5, self.make_divisible(scale * 160), self.make_divisible(scale * 960),
                    self.make_divisible(scale * 160), 'hard_swish', True, 1),
            )
            cls_ch_squeeze = 960
            self.conv2 = nn.Conv2d(self.make_divisible(scale * 160),
                self.make_divisible(scale * cls_ch_squeeze),
                kernel_size=1, stride=1, padding=0, bias=False)
        elif model_name == 'small':
            self.conv = nn.Sequential(
                Block(3, inplanes, self.make_divisible(scale * 16),
                    self.make_divisible(scale * 16), 'relu', True, 2),

                Block(3, self.make_divisible(scale * 16), self.make_divisible(scale * 72),
                    self.make_divisible(scale * 24), 'relu', False, 2),

                Block(3, self.make_divisible(scale * 24), self.make_divisible(scale * 88),
                    self.make_divisible(scale * 24), 'relu', False, 1),

                Block(5, self.make_divisible(scale * 24), self.make_divisible(scale * 96),
                    self.make_divisible(scale * 40), 'hard_swish', True, 2),

                Block(5, self.make_divisible(scale * 40), self.make_divisible(scale * 240),
                    self.make_divisible(scale * 40), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 40), self.make_divisible(scale * 240),
                    self.make_divisible(scale * 40), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 40), self.make_divisible(scale * 120),
                    self.make_divisible(scale * 48), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 48), self.make_divisible(scale * 144),
                    self.make_divisible(scale * 48), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 48), self.make_divisible(scale * 288),
                    self.make_divisible(scale * 96), 'hard_swish', True, 2),

                Block(5, self.make_divisible(scale * 96), self.make_divisible(scale * 576),
                    self.make_divisible(scale * 96), 'hard_swish', True, 1),

                Block(5, self.make_divisible(scale * 96), self.make_divisible(scale * 576),
                    self.make_divisible(scale * 96), 'hard_swish', True, 1),
            )
            cls_ch_squeeze = 576
            self.conv2 = nn.Conv2d(self.make_divisible(scale * 96),
                self.make_divisible(scale * cls_ch_squeeze),
                kernel_size=1, stride=1, padding=0, bias=False)
        else:
            raise NotImplementedError("mode[" + model_name +"_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, "supported scales are {} but input scale is {}".format(supported_scale, scale)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bn2 = nn.BatchNorm2d(self.make_divisible(scale*cls_ch_squeeze))
        self.hs2 = hswish()

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(feas[model_name][scale],num_classes)

        self.init_params()

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out1 = self.stage1(out)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out = self.pool1(self.hs2(self.bn2(self.conv2(out3))))
        '''head'''
        out = self.pool2(out)
        out = out.squeeze().unsqueeze(0) if out.shape[0] == 1 else out.squeeze()
        #out = out.view(8, -1)
        out = self.fc(out)
        return {1: out1, 2:out2, 3:out3}


def test():
    net = MobileNetV3(model_name = 'large', scale = 0.35)
    x = torch.randn(8,3,640,640)
    y = net(x)
    print(y.size())
