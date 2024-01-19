import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .convolutional_rnn import Conv2dGRU, Conv3dGRU
from networks.SE import *

# -----------------------------------------
# --------------- 3D UNet Start -----------
# -----------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=3, wf=5, padding=True,
                 norm='None', up_mode='upconv', residual=False):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.residual = residual

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf+i), padding, norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf+i), up_mode,
                                            padding, norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        input_ = x
        blocks = []

        # Contraction
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)

        # Expension
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        if self.residual:
            out = input_[:, [0], :, :, :] + self.last(x)
        else:
            out = self.last(x)

        return out

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()  # 32,32,32
        diff_y = (layer_height - target_size[0]) // 2  # floor division
        diff_x = (layer_width - target_size[1]) // 2
        diff_z = (layer_depth - target_size[2]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1]), diff_z:(diff_z + target_size[2])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
# -----------------------------------------
# --------------- 3D UNet End ----------------
# -----------------------------------------



# -----------------------------------------
# --------------- 3D DuRDN Start -----------
# -----------------------------------------
'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 4)
'''
class DuRDN(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None'):
        super(DuRDN, self).__init__()

        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        # decode
        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        # encode
        down1 = self.conv_in(x)  # [16, 64, 16, 16, 16]

        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)   # [16,64,16,16,16]
        down2 = F.avg_pool3d(SE1, 2)  # [16, 64, 8,8,8]

        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)
        down3 = F.avg_pool3d(SE2, 2)  # [16, 64, 4,4,4]

        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)  # [64, 4]

        # decode
        up2 = self.up2(SE3)  # [64, 8]

        RDB_up2 = self.RDB_up2(up2 + SE2)
        SE_up2 = self.SE_up2(RDB_up2)  # [64, 8]
        up1 = self.up1(SE_up2)  # [64, 16]

        RDB_up1 = self.RDB_up1(up1 + SE1) # [64, 16]
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)  # [16, 1, 16,16,16]

        return output


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out


# Make Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out
# -----------------------------------------
# --------------- 3D DuRDN End -----------
# -----------------------------------------






class Dis(nn.Module):
    def __init__(self, input_dim, n_layer=3, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        self.model = self._make_net(input_dim, ch, n_layer, norm, sn)

    def _make_net(self, input_dim, ch, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=4, stride=2, padding=1, norm='None', sn=sn)]
        tch = ch
        for i in range(n_layer-1):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=4, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        if sn:
            pass
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# GRU Residual Dense UNet
class GRURDSEUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=64, n_denselayer=6, growth_rate=32):
        super(GRURDSEUNet, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.SERDB1 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.SERDB2 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.SERDB3 = SERDB(n_filters * 1, n_denselayer, growth_rate)

        # configure conv GRU
        self.cgru = Conv2dGRU(in_channels=n_filters,  # Corresponds to input size
                              out_channels=n_filters,  # Corresponds to hidden size
                              kernel_size=(3, 3),  # Int or List[int]
                              num_layers=2,
                              bidirectional=False,
                              dilation=2, stride=2, dropout=0)

        # decode
        self.up3 = nn.ConvTranspose2d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SERDB4 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.up4 = nn.ConvTranspose2d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SERDB5 = SERDB(n_filters * 1, n_denselayer, growth_rate)

        self.conv2 = nn.Conv2d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, h):
        x = self.conv1(x)

        # encode
        SERDB1 = self.SERDB1(x)
        x = F.avg_pool2d(SERDB1, 2)

        SERDB2 = self.SERDB2(x)
        x = F.avg_pool2d(SERDB2, 2)

        SERDB3 = self.SERDB3(x)

        # cgru on latent volume
        SERDB3, h = self.cgru(SERDB3.unsqueeze(0), h)
        SERDB3 = SERDB3.squeeze(0)

        # decode
        up3 = self.up3(SERDB3)
        SERDB4 = self.SERDB4(up3 + SERDB2)

        up4 = self.up4(SERDB4)
        SERDB5 = self.SERDB5(up4 + SERDB1)

        output = self.conv2(SERDB5)

        return output, h


class RDSEUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=64, n_denselayer=6, growth_rate=32):
        super(RDSEUNet, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.SERDB1 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.SERDB2 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.SERDB3 = SERDB(n_filters * 1, n_denselayer, growth_rate)

        # decode
        self.up3 = nn.ConvTranspose2d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SERDB4 = SERDB(n_filters * 1, n_denselayer, growth_rate)
        self.up4 = nn.ConvTranspose2d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.SERDB5 = SERDB(n_filters * 1, n_denselayer, growth_rate)

        self.conv2 = nn.Conv2d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)

        # encode
        SERDB1 = self.SERDB1(x)
        x = F.avg_pool2d(SERDB1, 2)

        SERDB2 = self.SERDB2(x)
        x = F.avg_pool2d(SERDB2, 2)

        SERDB3 = self.SERDB3(x)

        # decode
        up3 = self.up3(SERDB3)
        SERDB4 = self.SERDB4(up3 + SERDB2)

        up4 = self.up4(SERDB4)
        SERDB5 = self.SERDB5(up4 + SERDB1)

        output = self.conv2(SERDB5)

        return output


# Squeeze and ExcitE Residual dense block (SERDB) architecture
class SERDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(SERDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        self.se = SELayer(nChannels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = self.se(out)
        out = out + x
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


##################################################################################
# Basic Functions
##################################################################################


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

##################################################################################
# Basic Blocks
##################################################################################


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            pass
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        elif norm == 'Batch':
            model += [nn.BatchNorm2d(n_out)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)



class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


if __name__ == '__main__':
    pass
