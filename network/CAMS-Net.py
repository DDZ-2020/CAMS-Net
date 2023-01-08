import torch.nn as nn
from torch.nn import functional as F
import torch
from LossAndEval import DiceLoss
import math

in_place = True
affine_par = True


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, has_bn=True, has_relu=True, weight_std=True):
        super(ConvBnRelu, self).__init__()
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(in_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU()

        self.conv = conv2d_3x3(in_planes=in_planes, out_planes=out_planes, kernel_size=ksize, stride=stride,
                               padding=pad, weight_std=weight_std)

    def forward(self, x):

        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        x = self.conv(x)

        return x

class _AMFS_sub(nn.Sequential):
    def __init__(self, input_num, output_num, dilation_rate, weight_std=True, drop_out=0.5):
        super(_AMFS_sub, self).__init__()
        self.se = SEBlock(input_num, r=16)

        self.bn = nn.BatchNorm2d(input_num)
        self.conv_d = conv2d_3x3(in_planes=input_num, out_planes=output_num, kernel_size=3, stride=1,
                                 padding=dilation_rate, dilation=dilation_rate, weight_std=weight_std)
        # nn.Conv2d(input_num, output_num, kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.relu = nn.ReLU(inplace=in_place)



        self.drop = nn.Dropout2d(drop_out)

    def forward(self, _input):
        # SE
        se = self.se(_input)

        # 空洞卷积
        x = self.bn(se)
        x = self.relu(x)
        x = self.conv_d(x)

        # out = self.drop(out)

        return x

class AMFSBlock(nn.Module):
    def __init__(self, num_features, weight_std=True, output_num=256, dropout0=0.5):
        super(AMFSBlock, self).__init__()

        self.dilate_1 = _AMFS_sub(input_num=num_features, output_num=output_num, dilation_rate=1, weight_std=weight_std)
        self.dilate_2 = _AMFS_sub(input_num=num_features, output_num=output_num, dilation_rate=2, weight_std=weight_std)
        self.dilate_6 = _AMFS_sub(input_num=num_features, output_num=output_num, dilation_rate=6, weight_std=weight_std)

        self.conv = ConvBnRelu(in_planes=3 * output_num, out_planes=num_features, ksize=1, stride=1, pad=0, weight_std=weight_std)

    def forward(self, x):
        d_1 = self.dilate_1(x)

        d_2 = self.dilate_2(x)

        d_6 = self.dilate_6(x)

        output = torch.cat((d_1, d_2, d_6), dim=1)

        output = self.conv(output)

        return output


class CASBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        # F_g为编码器特征
        # F_l为解码器特征
        # F_int为输出通道
        super(CASBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        r = 16
        if F_int <= 64:
            r = F_int // 8
        self.se = SEBlock(F_int, r=r)

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)

        psi_se = self.se(psi)
        psi = self.psi(psi_se)

        return x * psi


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)

        return y




# Pytorch
class Conv2d(nn.Conv2d):
    '''
    shape:
    input: (Batch_size, in_channels, H_in, W_in)
    output: ((Batch_size, out_channels, H_out, W_out))
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)

    # 权重标椎化
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv2d_3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, weight_std=True):
    '''3x3 convolution with padding'''
    if weight_std:
        return Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                         dilation=dilation, bias=bias)


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1, weight_std=True,
                 att=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std

        self.conv1 = conv2d_3x3(inplanes, planes, kernel_size=3, stride=stride, padding=1,
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place)
        self.bn1 = nn.BatchNorm2d(inplanes)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv2d_3x3(planes, planes, kernel_size=3, stride=1, padding=1, dilation=dilation * multi_grid,
                                bias=False, weight_std=self.weight_std)

        self.attblock = SEBlock(planes)
        self.att = att

        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.att:
            out = self.attblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet_resblock(nn.Module):
    def __init__(self, layers, num_classes=4, weight_std=False):
        super(unet_resblock, self).__init__()

        self.inplanes = 128
        self.weight_std = weight_std
        channels = [64, 128, 256, 512]

        self.conv1 = conv2d_3x3(3, channels[0], stride=1, weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, channels[0], channels[0], layers[0], stride=1)
        self.layer1 = self._make_layer(NoBottleneck, channels[0], channels[1], layers[1], stride=2)
        self.layer2 = self._make_layer(NoBottleneck, channels[1], channels[2], layers[2], stride=2)
        self.layer3 = self._make_layer(NoBottleneck, channels[2], channels[3], layers[3], stride=2)

        self.AMFSBlock = AMFSBlock(channels[3])

        self.fusionConv = nn.Sequential(
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=in_place),
            conv2d_3x3(channels[3], channels[2], kernel_size=3, padding=1, weight_std=self.weight_std),
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.d2_resb = self._make_layer(NoBottleneck, 2 * channels[2], channels[1], 1, stride=1)
        self.d3_resb = self._make_layer(NoBottleneck, 2 * channels[1], channels[0], 1, stride=1)
        self.d4_resb = self._make_layer(NoBottleneck, 2 * channels[0], channels[0], 1, stride=1)

        self.skip0_att = CASBlock(channels[0], channels[0], channels[0] // 2)
        self.skip1_att = CASBlock(channels[1], channels[1], channels[1] // 2)
        self.skip2_att = CASBlock(channels[2], channels[2], channels[2] // 2)

        self.map1 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            conv2d_3x3(channels[2], channels[0], kernel_size=3, padding=1, weight_std=self.weight_std),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )

        self.map2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            conv2d_3x3(channels[1], channels[0], kernel_size=3, padding=1, weight_std=self.weight_std),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )

        self.map3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv2d_3x3(channels[0], channels[0], kernel_size=3, padding=1, weight_std=self.weight_std),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=in_place),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )

        # 分类
        self.cls_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, input, mask):

        x = self.conv1(input)
        x = self.layer0(x)
        skip0_low = x

        x = self.layer1(x)
        skip1_low = x

        x = self.layer2(x)
        skip2_low = x

        x = self.layer3(x)
        x = self.ASPPBlock(x)
        x = self.fusionConv(x)

        x = self.upsamplex2(x)
        skip2_att = self.skip2_att(x, skip2_low)
        x = torch.cat((skip2_att, x), dim=1)
        d2 = self.d2_resb(x)

        x = self.upsamplex2(d2)
        skip1_att = self.skip1_att(x, skip1_low)
        x = torch.cat((skip1_att, x), dim=1)
        d3 = self.d3_resb(x)

        x = self.upsamplex2(d3)
        skip0_att = self.skip0_att(x, skip0_low)
        x = torch.cat((skip0_att, x), dim=1)
        d4 = self.d4_resb(x)

        d2 = self.map2(d2)
        d3 = self.map3(d3)
        cls_out = self.cls_conv(d4)

        loss_d2 = self.compute_multi_loss(d2, mask)
        loss_d3 = self.compute_multi_loss(d3, mask)
        loss_d4 = self.compute_multi_loss(cls_out, mask)
        self._loss = loss_d4 + 0.6 * loss_d3 + 0.4 * loss_d2

        return cls_out[:, 0, :, :], cls_out[:, 1, :, :], cls_out[:, 2, :, :], cls_out[:, 3, :, :]

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1, att=False):
        downsample = None

        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=in_place),
                conv2d_3x3(inplanes, planes, kernel_size=1, stride=stride, padding=0, weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1

        if stride == 1 and att:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std, att=att))
        else:
            layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))

        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std, att=att))
            else:
                layers.append(
                    block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                          weight_std=self.weight_std))

        return nn.Sequential(*layers)

    @property
    def loss(self):
        return self._loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1)
                # torch.nn.init.xavier_normal_(m.weight.data, gain=1)
                # torch.nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def compute_multi_loss(self, output, mask):
        self.criterion2 = DiceLoss()

        loss_mask = 0
        for i in range(len(mask)):
            loss_mask = loss_mask + self.criterion2(output[:, i, :, :].squeeze(), mask[i].squeeze())

        return loss_mask / len(mask)


def UNet_resblock(num_classes=4):
    print('using CAMS-Net')

    weight_std = True
    model = unet_resblock([1, 2, 2, 2], num_classes, weight_std)
    model._initialize_weights()

    return model
