import torch, cv2, math, sys, os, time, glob
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torchvision as tv
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from PIL import Image as im 
from google.colab import files

from torch import tensor
from torch.nn import Parameter
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

# Network Module
# ----------------------------------------
#               Conv2d Block
# ----------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, bias = False))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation, bias = False)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

class ResConv2dLayer(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride = 1, padding = 1, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True):
        super(ResConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.conv2d = nn.Sequential(
            Conv2dLayer(channels, channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn),
            Conv2dLayer(channels, channels, kernel_size, stride, padding, dilation, pad_type, activation = 'none', norm = 'none', sn = sn)
        )
    
    def forward(self, x):
        residual = x
        out = self.conv2d(x)
        out = out + residual
        return out

# ----------------------------------------
#            ConvLSTM2d Block
# ----------------------------------------
class ConvLSTM2d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)).cuda(),
                Variable(torch.zeros(state_size)).cuda()
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

# ----------------------------------------
#               Layer Norm
# ----------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# ----------------------------------------
#           Spectral Norm Block
# ----------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# ----------------------------------------
#             Attention Block
# ----------------------------------------
class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_FM, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)
        
        out = self.gamma * out
        return out, attention

class Self_Attn_C(nn.Module):
    """ Self attention Layer for Channel dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_C, self).__init__()
        self.chanel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.out_conv = nn.Conv2d(in_channels = in_dim // latent_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X c X c
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x c x c
        energy =  torch.bmm(proj_key, proj_query)
        # attention: B x c x c
        attention = self.softmax(energy)
        # proj_value is a convolution, B x c x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(batchsize, self.channel_latent, height, width)
        out = self.out_conv(out)
        
        out = self.gamma * out
        return out, attention

class ResAttnBlock(nn.Module):
    def __init__(self, in_dim, latent_dim = 8, pad_type = 'zero', norm = 'none'):
        super(ResAttnBlock, self).__init__()
        # Attention blocks
        self.attn_fm = Self_Attn_FM(in_dim, latent_dim)
        self.attn_c = Self_Attn_C(in_dim, latent_dim)
        # ResBlock
        self.res_gamma = nn.Parameter(torch.zeros(1))
        self.res_conv = Conv2dLayer(in_dim, in_dim, 3, 1, 1, pad_type = pad_type, norm = norm)

    def forward(self, x):
        attn_fm, attn_fm_map = self.attn_fm(x)
        attn_c, attn_c_map = self.attn_c(x)
        res_conv = self.res_gamma * self.res_conv(x)
        out = x + attn_fm + attn_c + res_conv
        return out

      
# Network
# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
# Generator contains 2 Auto-Encoders
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(3, 64, 7, 1, 3, pad_type = 'reflect', norm = 'none')
        self.E2 = Conv2dLayer(64, 64 * 2, 4, 2, 1, pad_type = 'reflect', norm = 'in')
        self.E3 = Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type = 'reflect', norm = 'in')
        self.E4 = Conv2dLayer(64 * 4, 64 * 8, 4, 2, 1, pad_type = 'reflect', norm = 'in')
        # Bottle neck
        self.BottleNeck = nn.Sequential(
            ResConv2dLayer(64 * 8, 3, 1, 1, pad_type = 'reflect', norm = 'in'),
            ResConv2dLayer(64 * 8, 3, 1, 1, pad_type = 'reflect', norm = 'in'),
            ResConv2dLayer(64 * 8, 3, 1, 1, pad_type = 'reflect', norm = 'in'),
            ResConv2dLayer(64 * 8, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        # Decoder
        self.D1 = TransposeConv2dLayer(64 * 8, 64 * 4, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2)
        self.D2 = TransposeConv2dLayer(64 * 8, 64 * 2, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2)
        self.D3 = TransposeConv2dLayer(64 * 4, 64 * 1, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2)
        self.D4 = Conv2dLayer(64 * 2, 3, 7, 1, 3, pad_type = 'reflect', norm = 'none', activation = 'tanh')
        # Sal Decoder
        self.SalDecoder = SalGenerator()

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 224 * 224
        E2 = self.E2(E1)                                        # out: batch * 128 * 112 * 112
        E3 = self.E3(E2)                                        # out: batch * 256 * 56 * 56
        E4 = self.E4(E3)                                        # out: batch * 512 * 28 * 28
        # Bottle neck
        E4 = self.BottleNeck(E4)                                # out: batch * 512 * 28 * 28
        # Decode the center code
        D1 = self.D1(E4)                                        # out: batch * 256 * 56 * 56
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 56 * 56
        D2 = self.D2(D1)                                        # out: batch * 128 * 112 * 112
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 112 * 112
        D3 = self.D3(D2)                                        # out: batch * 64 * 224 * 224
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 224 * 224
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256
        # Sal Decode
        #sal = self.SalDecoder(D1, D2, D3)

        return x

class SalGenerator(nn.Module):
    def __init__(self):
        super(SalGenerator, self).__init__()
        # Decoder 1
        self.D1 = nn.Sequential(
            TransposeConv2dLayer(64 * 8, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2),
            Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in'),
            TransposeConv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2),
            Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        self.D2 = nn.Sequential(
            TransposeConv2dLayer(64 * 4, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in', scale_factor = 2),
            Conv2dLayer(64, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        self.D3 = Conv2dLayer(64 * 2, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        # Decoder 2
        self.D4 = nn.Sequential(
            Conv2dLayer(64 * 3, 64, 3, 1, 1, pad_type = 'reflect', norm = 'in'),
            Conv2dLayer(64, 1, 7, 1, 3, pad_type = 'reflect', norm = 'none', activation = 'sigmoid')
        )

    def forward(self, D1, D2, D3):
        D1 = self.D1(D1)
        D2 = self.D2(D2)
        D3 = self.D3(D3)
        D4 = torch.cat((D1, D2, D3), 1)
        D4 = self.D4(D4)

        return D4

# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = nn.Sequential(
            Conv2dLayer(3, 64, 1, 1, 0, pad_type = 'reflect', norm = 'none'),
            Conv2dLayer(64, 64, 7, 1, 3, pad_type = 'reflect', norm = 'in')
        )
        self.block2 = nn.Sequential(
            Conv2dLayer(64 , 64 * 2, 4, 2, 1, pad_type = 'reflect', norm = 'in'),
            Conv2dLayer(64 * 2, 64 * 2, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        self.block3 = nn.Sequential(
            Conv2dLayer(64 * 2, 64 * 4, 4, 2, 1, pad_type = 'reflect', norm = 'in'),
            Conv2dLayer(64 * 4, 64 * 4, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        self.block4 = nn.Sequential(
            Conv2dLayer(64 * 4, 64 * 8, 4, 2, 1, pad_type = 'reflect', norm = 'in'),
            Conv2dLayer(64 * 8, 64 * 8, 3, 1, 1, pad_type = 'reflect', norm = 'in')
        )
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(64 * 8, 64 * 4, 4, 1, 1, pad_type = 'reflect', norm = 'in')
        self.final2 = Conv2dLayer(64 * 4, 1, 4, 1, 1, pad_type = 'reflect', norm = 'none', activation = 'none')

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        # img_A: input; img_B: output
        x = img_A
        #x = torch.cat((img_A, img_B), 1)                        # out: batch * 7 * 256 * 256
        block1 = self.block1(x)                                 # out: batch * 64 * 256 * 256
        block2 = self.block2(block1)                            # out: batch * 128 * 128 * 128
        block3 = self.block3(block2)                            # out: batch * 256 * 64 * 64
        x = self.block4(block3)                                 # out: batch * 512 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x

# ----------------------------------------
#             Perceptual Net
# ----------------------------------------
# For perceptual loss
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
      
def create_generator():
    # Initialize the network
    generator = Generator()
    # Init the network
    weights_init(generator, init_type = 'normal', init_gain = 0.02)
    print('Generator is created!')
    return generator

def create_discriminator():
    # Initialize the network
    discriminator = PatchDiscriminator70()
    # Init the network
    weights_init(discriminator, init_type = 'normal', init_gain = 0.02)
    print('Discriminator is created!')
    return discriminator
    
def create_perceptualnet():
    # Initialize the network
    perceptualnet = PerceptualNet()
    vgg16 = tv.models.vgg16(pretrained = True)
    # Init the network
    load_dict(perceptualnet, vgg16)
    print('PerceptualNet is created!')
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet
    
def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net.state_dict()
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def train_model(train_dl, generator, discriminator):
    lr = 0.0003
    d_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optim = torch.optim.Adam(generator.parameters(), lr=lr)
    for epoch in tqdm(range(1,num_epoch+1)):
        print('Training Discriminator')
        discriminator.train()

        for data in train_dl:
            true_inputs, true_labels = data      
            true_inputs = true_inputs.cuda()

            for epoch in range(5):
                fake_inputs = generator(true_inputs)
                true_target = discriminator(true_inputs)
                fake_target = discriminator(fake_inputs)
                loss = criterion(fake_inputs, true_inputs)
                loss.backward()
                d.optim.step()

            generator.train()
            print('Training Generator')
            g_optim.zero_grad()
            fake_inputs = generator(true_inputs)
            gan_loss = criterion(fake_inputs, true_inputs)

            fake_percep_feature = perceptualnet(fake_target)
            true_percep_feature = perceptualnet(true_target)
            perceptual_loss = criterion(fake_percep_feature, true_percep_feature)
            loss = perceptual_loss + gan_loss

            loss.backward()
            g_optim.step()
            
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

batchSize = 32; num_epoch=30
trainpath = 'drive/MyDrive/5th Yr Project/TestData/FiveK'
testpath = 'drive/MyDrive/5th Yr Project/TestData/Kodak'

trainset = datasets.ImageFolder(root=trainpath,transform=preprocess)
trainloader = DataLoader(dataset=trainset,batch_size=batchSize, shuffle=True)
testset = datasets.ImageFolder(root=testpath,transform=preprocess)
testloader = DataLoader(dataset=testset,batch_size=24, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = create_generator()
discriminator = create_discriminator()
perceptualnet = create_perceptualnet()

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    criterion = torch.nn.L1Loss().cuda()
else:
    criterion = torch.nn.BCELoss()

train_model(trainloader, generator, discriminator)

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cpu()
        outputs = generator(images)
        outputs = outputs.cpu()
        imshow(torchvision.utils.make_grid(outputs))
