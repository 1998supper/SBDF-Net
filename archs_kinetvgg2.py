# from torchvision.utils import save_image
# import os
# from myarchs.torch_components import BaseConv, MDPBlock, BDPBlock, DownBlock
# from nets.diversebranchblock import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# from archs_axialnet import *
# from archs_TransCNN import InvertedResidual,Upsample,DoubleDownsample
from archs_unet import asa_layer

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class DSConv5x5(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv5x5, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=5, s=stride, p=2*dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class asa_layer(nn.Module):
    def __init__(self, channel, groups=16):
        super().__init__()
        self.groups = groups

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // groups, 1, 1))

        self.cbias = nn.Parameter(torch.ones(1, channel // groups, 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // groups, 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel //  groups, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // groups, channel //  groups)
        self.oneconv1 = nn.Conv2d(channel*3, channel, 1)

        # bi-linear modelling for both
        self.W_g = Conv(channel, channel, 1, bn=True, relu=False)
        self.W_x = Conv(channel, channel, 1, bn=True, relu=False)
        self.W = Conv(channel, channel, 3, bn=True, relu=True)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x, y):
        b, c, h, w = x.shape

        # bilinear pooling
        W_g = self.W_g(x)
        W_x = self.W_x(y)
        bp = self.W(W_g * W_x)
        bp = bp.reshape(b * self.groups, -1, h, w)

        x = x.reshape(b * self.groups, -1, h, w)
        y = y.reshape(b * self.groups, -1, h, w)
        x_0 = x
        x_1 = y

        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs, bp], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 3)
        out = self.oneconv1(out)
        return out

class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            norm_layer(out_dim)
        )
        self.drop = nn.Dropout2d(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x

class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        self.residual = nn.Conv2d(in_dim+out_dim, out_dim*2, 1)
        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(in_dim)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x1 = self.norm1(self.conv(x))
        x2 = self.norm2(F.interpolate(x, scale_factor=(2, 2), mode='bilinear'))
        x = self.act(self.residual(torch.cat([x1 , x2],1)))
        return x

class DoubleDownsample(nn.Module):
    def __init__(self, in_dim, out_dim, outsize, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(outsize)
        self.residual = nn.Conv2d(in_dim*2, out_dim, 1)
        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(out_dim)
        self.act = act_layer(inplace=True)


    def forward(self, x):
        x1 = self.norm1(self.conv(x))
        x2 = self.norm2(self.residual(torch.cat([self.pool(x),self.avgpool(x)],1)))
        x = self.act(x1 + x2)
        return x

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)
class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = convbnrelu(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = convbnrelu(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = self.out(torch.cat([x, feat1, feat2, feat3, feat4], dim=1)) + x

        return x

class dbunetinkifdatrnewfddnbakmmfalre30(nn.Module):
    def __init__(self):
        super(dbunetinkifdatrnewfddnbakmmfalre30, self).__init__()

        dims = [64, 128, 256, 512, 1024]
        kernel_sizes = [5, 3, 5, 3]
        norm_layer = nn.BatchNorm2d
        act_layer = nn.SiLU
        inputsizes = 48
        # inputsizes = 64
        self.encoder1 = InvertedResidual(3, hidden_dim=3 * 4, out_dim=32, kernel_size=kernel_sizes[1], drop=0.,act_layer=act_layer, norm_layer=norm_layer)
        self.encoder2 = InvertedResidual(dims[0], hidden_dim=dims[0] * 4, out_dim=dims[0], kernel_size=kernel_sizes[0],drop=0., act_layer=act_layer, norm_layer=norm_layer)
        self.encoder3 = InvertedResidual(dims[1], hidden_dim=dims[1] * 4, out_dim=dims[1], kernel_size=kernel_sizes[1],drop=0., act_layer=act_layer, norm_layer=norm_layer)
        self.encoder4 = InvertedResidual(dims[2], hidden_dim=dims[2] * 4, out_dim=dims[2], kernel_size=kernel_sizes[2],drop=0., act_layer=act_layer, norm_layer=norm_layer)
        self.encoder5 = InvertedResidual(dims[3], hidden_dim=dims[3] * 4, out_dim=dims[3], kernel_size=kernel_sizes[3],drop=0., act_layer=act_layer, norm_layer=norm_layer)
        self.ds1 = DoubleDownsample(32, dims[0], norm_layer=norm_layer, outsize=inputsizes)
        self.ds2 = DoubleDownsample(dims[0], dims[1], norm_layer=norm_layer, outsize=inputsizes // 2)
        self.ds3 = DoubleDownsample(dims[1], dims[2], norm_layer=norm_layer, outsize=inputsizes // 4)
        self.ds4 = DoubleDownsample(dims[2], dims[3], norm_layer=norm_layer, outsize=inputsizes // 8)
        self.ds5 = DoubleDownsample(dims[3], dims[4], norm_layer=norm_layer, outsize=inputsizes // 8)
        self.ds22 = DoubleDownsample(dims[0], dims[1], norm_layer=norm_layer, outsize=inputsizes * 2)
        self.ds33 = DoubleDownsample(dims[1], dims[2], norm_layer=norm_layer, outsize=inputsizes )
        self.ds44 = DoubleDownsample(dims[2], dims[3], norm_layer=norm_layer, outsize=inputsizes // 2)
        self.ds55 = DoubleDownsample(dims[3], dims[4], norm_layer=norm_layer, outsize=inputsizes // 4)

        base_chan=32
        # base_chan=64
        self.encoder11 = DSConv3x3(32, 64, dilation=1)
        self.encoder22 = DSConv3x3(64, 128, dilation=1)
        self.encoder33 = DSConv3x3(128, 256, dilation=2)
        self.encoder44 = DSConv3x3(256, 512, dilation=2)
        self.encoder55 = DSConv5x5(512, 512, dilation=2)
        self.encoder66 = DSConv5x5(512, 1024, dilation=2)

        self.ddecoder12 = InvertedResidual(1024, hidden_dim=512 * 4, out_dim=1024, kernel_size=kernel_sizes[1], drop=0.,
                                           act_layer=act_layer, norm_layer=norm_layer)
        self.ddecoder22 = InvertedResidual(1024, hidden_dim=512 * 4, out_dim=512, kernel_size=kernel_sizes[0], drop=0.,
                                           act_layer=act_layer, norm_layer=norm_layer)
        self.ddecoder31 = InvertedResidual(512, hidden_dim=256 * 4, out_dim=256, kernel_size=kernel_sizes[1], drop=0.,
                                           act_layer=act_layer, norm_layer=norm_layer)
        self.ddecoder41 = InvertedResidual(256, hidden_dim=128 * 4, out_dim=128, kernel_size=kernel_sizes[0], drop=0.,
                                           act_layer=act_layer, norm_layer=norm_layer)
        self.ddecoder51 = InvertedResidual(128, hidden_dim=64 * 4, out_dim=64, kernel_size=kernel_sizes[1], drop=0.,
                                           act_layer=act_layer, norm_layer=norm_layer)

        self.upds3 = Upsample(1024, 512, norm_layer=norm_layer)
        self.upds4 = Upsample(512, 256, norm_layer=norm_layer)
        self.upds5 = Upsample(256, 128, norm_layer=norm_layer)
        self.upds6 = Upsample(128, 64, norm_layer=norm_layer)

        self.upds7 = Upsample(32, 16, norm_layer=norm_layer)
        self.upds8 = nn.Sequential(Upsample(dims[0], 32, norm_layer=norm_layer),Upsample(dims[0], 32, norm_layer=norm_layer))
        self.upds9 = nn.Sequential(Upsample(dims[1], dims[1]//2, norm_layer=norm_layer),Upsample(dims[1], dims[1]//2, norm_layer=norm_layer))
        self.upds10 = nn.Sequential(Upsample(dims[2], dims[2]//2, norm_layer=norm_layer),Upsample(dims[2], dims[2]//2, norm_layer=norm_layer))
        self.upds11 = nn.Sequential(Upsample(dims[3], dims[3]//2, norm_layer=norm_layer),Upsample(dims[3], dims[3]//2, norm_layer=norm_layer))

        self.de5_bn = nn.BatchNorm2d(1024)
        self.de4_bn = nn.BatchNorm2d(512)
        self.de3_bn = nn.BatchNorm2d(256)
        self.de2_bn = nn.BatchNorm2d(128)
        self.de1_bn = nn.BatchNorm2d(64)
        self.final5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.ec5 = nn.Conv2d(1024, 512, kernel_size=1)
        self.final4 = nn.Conv2d(512, 1, kernel_size=1)
        self.final3 = nn.Conv2d(256, 1, kernel_size=1)
        self.final2 = nn.Conv2d(128, 1, kernel_size=1)
        self.final1 = nn.Conv2d(64, 1, kernel_size=1)

        self.pyramid_pooling2 = PyramidPooling(512, 512)
        self.pyramid_pooling4 = PyramidPooling(1024, 1024)

        self.oneconv1 = nn.Conv2d(64,128,1)
        self.oneconv2 = nn.Conv2d(32,64,1)
        self.oneconv3 = nn.Conv2d(128,256,1)
        self.oneconv4 = nn.Conv2d(256,512,1)
        self.oneconv5 = nn.Conv2d(512,1024,1)
        self.oneconv6 = nn.Conv2d(32,128,1)
        self.oneconv7 = nn.Conv2d(128, 64, 1)
        self.oneconv8 = nn.Conv2d(256, 128, 1)
        self.oneconv9 = nn.Conv2d(512, 256, 1)
        self.oneconv10 = nn.Conv2d(1024, 512, 1)

        self.endconv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.final = nn.Conv2d(32, 1, 1)
        self.finala = nn.Conv2d(16, 1, 1)
        self.finalb = nn.Conv2d(64, 1, 1)
        self.finalc = nn.Conv2d(128, 1, 1)
        self.soft = nn.Softmax(dim=1)
        self.si = nn.SiLU(inplace=True)
        self.up_c0 = asa_layer(1024)
        self.up_c1 = asa_layer(512)
        self.up_c2 = asa_layer(256)
        self.up_c3 = asa_layer(128)
        self.up_c4 = asa_layer(64)


    def forward(self, x):

        out = self.encoder1(x)
        t0 = out #32
        out = self.encoder2(self.ds1(out))
        t1 = out

        out1 = self.up_c4(self.upds8(t1),self.encoder11(self.upds7(t0)))
        out1 = self.ds22(out1)

        t0 = self.oneconv2(t0)
        out1 = torch.add(t0*self.oneconv7(out1),t0)
        out1 = self.encoder22(out1)

        out = self.encoder3(self.ds2(out))
        t2 = out
        out1 = self.up_c3(self.upds9(t2),out1)
        t0 = out1
        out1 = self.ds33(out1)
        t1 = self.oneconv1(t1)
        out1 = torch.add(t1*self.oneconv8(out1),t1)
        out1 = self.encoder33(out1)

        out = self.encoder4(self.ds3(out))
        t3 = out
        out1 = self.up_c2(self.upds10(t3), out1)
        t1 = out1
        out1 = self.ds44(out1)
        t2 = self.oneconv3(t2)
        out1 = torch.add( t2*self.oneconv9(out1) , t2)
        out1 = self.encoder44(out1)

        out = self.encoder5(self.ds4(out))
        out = self.pyramid_pooling2(out)
        out1 = self.up_c1(self.upds11(out), out1)
        t2 = out1
        out1 = self.ds55(out1)
        t3 =  self.oneconv4(t3)
        out1 = torch.add( t3 * self.oneconv10(out1), t3)
        out1 = self.encoder55(out1)
        t3 = out1

        out1 = self.ds5(out1)
        out1 = self.oneconv10(out1)
        out1 = torch.add(out * out1, out)
        out1 = self.encoder66(out1)
        out1 = self.pyramid_pooling4(out1)

        output4 = self.ddecoder12(out1)
        out = self.upds3(output4)
        t3 = self.oneconv5(t3)
        out = torch.add(t3 * out, t3)
        output3 = self.ddecoder22(out)
        out = self.upds4(output3)
        out = torch.add(t2 * out, t2)
        output2 = self.ddecoder31(out)
        out = self.upds5(output2)
        out = torch.add(t1 * out, t1)
        output1 = self.ddecoder41(out)
        out = self.upds6(output1)
        out = torch.add(t0 * out, t0)
        out = self.ddecoder51(out)

        out = self.finalb(out)

        output4 = self.final4(self.ec5(self.si(self.de5_bn(F.interpolate(output4, size=(96, 96), mode='bilinear')))))
        output3 = self.final4(self.si(self.de4_bn(F.interpolate(output3, size=(96, 96), mode='bilinear'))))
        output2 = self.final3(self.si(self.de3_bn(F.interpolate(output2, size=(96, 96), mode='bilinear'))))
        output1 = self.final2(self.si(self.de2_bn(F.interpolate(output1, size=(96, 96), mode='bilinear'))))

        return out, output1, output2, output3,output4

if __name__ == "__main__":
    model = dbunetinkifdatrnewfddnbakmmfalre30()
    x = torch.randn((2, 3, 96, 96))
    out = model(x)
    for o in out:
        print(o.shape)