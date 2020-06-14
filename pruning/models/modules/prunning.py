

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
from scipy.optimize import root_scalar




class Conv2dStats(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dStats, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.fullName = ''

        self.gradOutputSparsity = 0
        self.gradOutputTau = 0
        self.gradOutputMinusTau = 0
        self.elems = 0

    def forward(self, input):
        output = super(Conv2dStats, self).forward(input)

        return output





class ZeroBN(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,track_running_stats=True):
        super(ZeroBN, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.fullName = ''

        self.final_tau = Variable(torch.tensor([0]).cuda())

        self.k = 0
        self.sigma = 0
        self.regularAngle = 0
        self.stochastic = True
        self.preserve_cosine = False
        self.max_sparsity = 0
        self.min_cos_sim = False
        self.max_cos_sim = False
        self.cos_sim_min = 0
        self.cos_sim = 0


    def forward(self, input):

        if self.training and self.final_tau > 0:
            input = BNPrun.apply(input, self.final_tau)
        output = super(ZeroBN, self).forward(input)

        return output



class BNPrun(Function):

    @staticmethod
    def forward(ctx, x, final_tau):
        ctx.save_for_backward(final_tau)
        return x

    @staticmethod
    def backward(ctx, grad_output):

        final_tau = ctx.saved_tensors
        if final_tau != 0:


            rand = final_tau * torch.rand(grad_output.shape, device="cuda", dtype=torch.float32)

            grad_abs = grad_output.abs()
            #
            grad = torch.where(grad_abs < final_tau, final_tau * torch.sign(grad_output), grad_output)
            grad = torch.where(grad_abs < rand, torch.tensor([0], device="cuda", dtype=torch.float32), grad)

        else:
            grad = grad_output



        return grad, None



