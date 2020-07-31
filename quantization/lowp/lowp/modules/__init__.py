import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from lowp.functional import *


def _fp8_args(**kwargs):
    prefix = kwargs.pop('prefix', '')
    _args = dict(exp_width=5, man_width=2, exp_bias=15,
                 roundingMode=1, lfsrVal=1)
    pass_args = {}
    for arg in _args:
        pass_args[arg] = kwargs.pop(prefix + arg, _args[arg])
    return kwargs, pass_args


class Conv1d_BF16(nn.Conv1d):
    """docstring for Conv2d_BF16."""

    def __init__(self, *args, **kwargs):
        super(Conv1d_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_bf16(input)
        weight = truncate_bf16(weight)
        if bias is not None:
            bias = truncate_bf16(bias)
        output = F.conv1d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return truncate_grad_bf16(output)


class Embedding_BF16(nn.Embedding):

    def __init__(self, *args, **kwargs):
        print('using bf16 embedding')
        super(Embedding_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        output = F.embedding(input, weight)
        return truncate_grad_bf16(output)


class Conv2d_FP8(nn.Conv2d):
    """docstring for Conv2d_FP8."""

    def __init__(self, *args, **kwargs):
        kwargs, self.fp8_args = _fp8_args(**kwargs)
        kwargs, self.fp8_grad_args = _fp8_args(prefix='grad_', **kwargs)    
        print(kwargs)    
        super(Conv2d_FP8, self).__init__(*args, **kwargs)


    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_fp8(input, **self.fp8_args)
        weight = truncate_fp8(weight, **self.fp8_args)
        if bias is not None:
            bias = truncate_fp8(bias, **self.fp8_args)
        output = F.conv2d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return truncate_grad_fp8(output, **self.fp8_grad_args)


class Linear_FP8(nn.Linear):
    """docstring for Linear_FP8."""

    def __init__(self, *args, **kwargs):
        print('using fp8 linear')
        kwargs, self.fp8_args = _fp8_args(**kwargs)
        kwargs, self.fp8_grad_args = _fp8_args(prefix='grad_', **kwargs)               
        super(Linear_FP8, self).__init__(*args, **kwargs)
 

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_fp8(input, **self.fp8_args)
        weight = truncate_fp8(weight, **self.fp8_args)
        if bias is not None:
            bias = truncate_fp8(bias, **self.fp8_args)
        output = F.linear(input, weight, bias)
        return truncate_grad_fp8(output, **self.fp8_grad_args)


class Conv2d_BF16(nn.Conv2d):
    """docstring for Conv2d_BF16."""

    def __init__(self, *args, **kwargs):
        super(Conv2d_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_bf16(input)
        weight = truncate_bf16(weight)
        if bias is not None:
            bias = truncate_bf16(bias)
        output = F.conv2d(input, weight, bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return truncate_grad_bf16(output)


class Linear_BF16(nn.Linear):
    """docstring for Linear_BF16."""

    def __init__(self, *args, **kwargs):
        print('using bf16 linear')
        super(Linear_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_bf16(input)
        weight = truncate_bf16(weight)
        if bias is not None:
            bias = truncate_bf16(bias)
        output = F.linear(input, weight, bias)
        return truncate_grad_bf16(output)


class AvgPool2d_BF16(nn.AvgPool2d):
    """docstring for AvgPool2d_BF16."""

    def __init__(self, *args, **kwargs):
        # print('using bf16 avg_pool')
        super(AvgPool2d_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        input = truncate_bf16(input)
        output = F.avg_pool2d(input, self.kernel_size, self.stride,
                              self.padding, self.ceil_mode, self.count_include_pad)
        return truncate_grad_bf16(output)


class BatchNorm2d_BF16(nn.BatchNorm2d):
    """docstring for BatchNorm2d_BF16."""

    def __init__(self, *args, **kwargs):
        # print('using bf16 bn')
        super(BatchNorm2d_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        running_mean = self.running_mean
        running_var = self.running_var
        input = truncate_bf16(input)
        if weight is not None:
            weight = truncate_bf16(weight)
        if bias is not None:
            bias = truncate_bf16(bias)
        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=self.training, momentum=self.momentum, eps=self.eps)
        return truncate_grad_bf16(output)


class LayerNorm_BF16(nn.LayerNorm):
    """docstring for BatchNorm2d_BF16."""

    def __init__(self, *args, **kwargs):
        print('using bf16 layer norm')
        super(LayerNorm_BF16, self).__init__(*args, **kwargs)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        input = truncate_bf16(input)
        if weight is not None:
            weight = truncate_bf16(weight)
        if bias is not None:
            bias = truncate_bf16(bias)
        output = F.layer_norm(input, self.normalized_shape,
                              weight, bias, self.eps)
        return truncate_grad_bf16(output)



