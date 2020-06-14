from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
from itertools import count
import numpy as np
import pickle
import time
from torch.autograd import Variable







def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)

def calculate_qparams(x, num_bits, flatten_dims = (1, -1), reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1:
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)
        if reduce_dim is not None:
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]
        range_values = max_values - min_values
        scale = (range_values) / (2. ** num_bits - 1.)

        return {'max': max_values, 'zero_point':min_values,'scale':scale}

class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, inplace=False):
        output = input.clone()


        zero_point = qparams['zero_point']
        qmin = 0
        qmax =  2. ** num_bits - 1.

        with torch.no_grad():
            output.add_(qmin * qparams['scale'] - zero_point).div_(qparams['scale'])

            output.clamp_(qmin, qmax).round_()


        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None, None





class ConvCat(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvCat, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.actBitwidth = 8
        self.num_bits_weight = 8
        # self.quantize_input = QuantMeasure(
        #     self.num_bits, shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))


        self.register_buffer('sparsity', torch.zeros(1))
        self.register_buffer('elems', torch.zeros(1))
        # self.softEntropy = torch.zeros(1).cuda()
        # self.softEntropy.requires_grad = True
        #self.register_buffer('softEntropy', torch.zeros(1))
        #self.softEntropy.requires_grad = True
        self.register_buffer('entropy', torch.zeros(1))

        self.fullName = ''
        #change this parameters
        self.mode = 'entropy'


    def forward(self, input):
        N, C, H, W = input.shape
        if C > 3:
            qparams = calculate_qparams(input, num_bits=self.actBitwidth)

            qinput = UniformQuantize().apply(input, self.actBitwidth, qparams)
            # dequantize
            qinput.mul_(qparams['scale']).add_(
                qparams['zero_point'])  # dequantize

        else:
            qinput = input #don't quantize input image


        # qinput = self.quantize_input(input)

        weight_qparams = calculate_qparams(
            self.weight, num_bits=self.num_bits_weight, flatten_dims=(1, -1), reduce_dim=None)
        qweight = UniformQuantize().apply(self.weight, self.num_bits_weight, weight_qparams)

        if self.elems == 0:
            self.elems.mul_(0).add_(torch.numel(self.weight))


        # if self.training:
        #
        #
        #
        #
        #     if self.mode == 'entropy':
        #         #self.softEntropy.detach()
        #         self.softEntropy = self.soft_entropy(subset, min=weight_qparams['zero_point'], max=weight_qparams['max'],
        #                            bits=self.num_bits_weight,
        #                            temp=-10)
        #     elif self.mode == 'compression':
        #         self.softEntropy = input.norm(p=1) / input.norm(p=2)
        #
        if not self.training:
            self.entropy.mul_(0).add_(self.shannon_entropy(qweight.detach(), bits=self.actBitwidth))

        # dequantize
        qweight.mul_(weight_qparams['scale']).add_(
            weight_qparams['zero_point'])  # dequantize

        self.sparsity.mul_(0).add_((torch.numel((qweight.view(-1) == 0).nonzero()) / torch.numel(qweight)))

        output = F.conv2d(input,self.weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)



        return output

    def shannon_entropy(self, x, base=2, bits=8):
        #   pk = torch.bincount(torch.round(x).long().flatten())

        pk = torch.histc(x, max=base**bits - 1, min=0, bins=base ** bits)

        pk = pk.float() / torch.sum(pk).float()

        pk[pk == 0] = 1  # HACK
        vec = -pk * torch.log(pk)
        return torch.sum(vec) / np.log(base)

    def soft_entropy(self, x, min, max, bits=8, temp=-10):
        if torch.numel(torch.unique(x)) == 1:
            return 0
        bins = int(2 ** bits)
        centers = torch.linspace(0, bins-1, bins).cuda()
        # act_scale = (2 ** bits - 1) / (max - min)
        #
        #      x = x.contiguous().view(-1)
        # x = (x - min) * act_scale

        x = (x.repeat(bins, 1).t() - centers) ** 2

        x = temp * x
        x = F.softmax(x, 1, _stacklevel=5)
        x = torch.sum(x, dim=0) / x.shape[0]
        x[x == 0] = 1  # hack
        x = -x * torch.log(x)
        return torch.sum(x) / np.log(2)



class ReLuCat(nn.ReLU):
    def __init__(self, inplace=False):
        super(ReLuCat, self).__init__(inplace)
        self.inplace = inplace

        self.actBitwidth = 8



        self.clip = False

        self.register_buffer('sparsity', torch.zeros(1))
        self.register_buffer('elems', torch.zeros(1))

    #    self.softEntropy = nn.Parameter(torch.Tensor(1))
    #    self.register_buffer('softEntropy', torch.zeros(1))

        self.register_buffer('entropy', torch.zeros(1))

        self.fullName = ''
        #change this parameters
        self.mode = 'entropy'

    def forward(self, input):
        N, C, H, W = input.shape  # N x C x H x W
        if self.elems == 0:
            self.elems.mul_(0).add_(N*C*H*W)



        output = super(ReLuCat, self).forward(input)

        qparams = calculate_qparams(output, num_bits=self.actBitwidth)

        output = UniformQuantize().apply(output, self.actBitwidth, qparams)

        if self.training:

            subset = output[0,:,:,:].view(-1)

            idx = torch.randperm(C * H * W)[:1000]

            subset = subset[idx]

            if self.mode == 'entropy':
                self.softEntropy = (self.soft_entropy(subset, min = qparams['zero_point'], max = qparams['max'] ,bits = self.actBitwidth,
                                             temp=-10))
            elif self.mode == 'compression':
                self.softEntropy = input.norm(p=1) / input.norm(p=2)

        self.entropy.mul_(0).add_(self.shannon_entropy(output.detach(),bits = self.actBitwidth))

        # dequantize
        output.mul_(qparams['scale']).add_(
            qparams['zero_point'])  # dequantize

        self.sparsity.mul_(0).add_((torch.numel((output.view(-1) == 0).nonzero()) / torch.numel(output)))

        return output




    def shannon_entropy(self, x, base=2,bits = 8):


     #   pk = torch.bincount(torch.round(x).long().flatten())

        pk = torch.histc(x, max  = torch.max(x).item(), min = torch.min(x).item(), bins = base**bits)

        pk = pk.float() / torch.sum(pk).float()

        pk[pk == 0] = 1  # HACK
        vec = -pk * torch.log(pk)
        return torch.sum(vec) / np.log(base)


    def soft_entropy(self,x, min,max, bits = 8, temp = -10):
        if max == min:
            return 0
        bins = int(2 ** bits)
        centers = torch.linspace(0, bins-1, bins).cuda()
        # act_scale = (2 ** bits - 1) / (max - min)
        #
        #      x = x.contiguous().view(-1)
        # x = (x - min) * act_scale

        x = (x.repeat(bins, 1).t() - centers) ** 2



        x = temp * x
        x = F.softmax(x, 1, _stacklevel=5)
        x = torch.sum(x, dim=0) / x.shape[0]
        x[x == 0] = 1  # hack
        x = -x * torch.log(x)
        return torch.sum(x) / np.log(2)


