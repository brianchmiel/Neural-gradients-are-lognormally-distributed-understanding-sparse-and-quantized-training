import torch
from torch.autograd.function import Function
from scipy.optimize import root_scalar

class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.scale * grad_output
        return grad_input, None


def scale_grad(x, scale):
    return ScaleGrad().apply(x, scale)

def negate_grad(x):
    return scale_grad(x, -1)


def equationForward( alpha, sparsity,b):

    fun = 1- sparsity + b*torch.exp(-alpha / b)/alpha - (b/alpha)
    deriative = torch.exp(-alpha / b) * (b*(torch.exp(-alpha / b) - 1) - alpha) / (alpha**2)

    return fun # , deriative


def calcAlpha(b,sparsity):

    guess = torch.tensor([0.001], device="cuda", dtype=torch.float)
    bracket = [torch.tensor([-3], device="cuda", dtype=torch.float),
               torch.tensor([300], device="cuda", dtype=torch.float)]

    if sparsity > 0:
        sol = root_scalar(equationForward, x0=guess, bracket=bracket, args=(sparsity, b))
        return torch.tensor([sol.root], device="cuda", dtype=torch.float)
    else:
        return torch.tensor([0], device="cuda", dtype=torch.float)
