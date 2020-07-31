import lowp._functions as f
import torch

# RNE
BF16_DEFAULT_ROUNDING_MODE=3

def quantemu(input, mode, inplace=False):
    return f.QuantEmu().apply(input.contiguous(), mode, inplace)


def truncate_bf16(input, inplace=False, roundingMode=BF16_DEFAULT_ROUNDING_MODE):
    return f.TruncateBF16().apply(input.contiguous(), inplace, roundingMode)


def truncate_grad_bf16(input, roundingMode=BF16_DEFAULT_ROUNDING_MODE, debugStr = None):
    return f.TruncateGradBF16().apply(input.contiguous(), roundingMode, debugStr)


def truncate_fp8(input, inplace=False, exp_width=5, man_width=2, exp_bias=15, roundingMode=1, lfsrVal=1):
    is_half = False
    if input.dtype == torch.half:
        assert not inplace
        input = input.float()
        is_half = True
    out = f.TruncateFP8().apply(input.contiguous(), inplace, exp_width,
                                man_width, exp_bias, roundingMode, lfsrVal)
    if is_half:
        out = out.half()
    return out


def truncate_grad_fp8(input, exp_width=5, man_width=2, exp_bias=7, roundingMode=1, lfsrVal=1):
    return f.TruncateGradFP8().apply(input, exp_width, man_width, exp_bias, roundingMode, lfsrVal)


def bmm_bf16(x, y):
    return truncate_grad_bf16(torch.bmm(truncate_bf16(x), truncate_bf16(y)))


def matmul_bf16(x, y):
    return truncate_grad_bf16(torch.matmul(truncate_bf16(x), truncate_bf16(y)))


def add_bf16(*kargs):
    bf16_args = []
    for arg in kargs:
        bf16_args.append(truncate_bf16(arg))
    return truncate_grad_bf16(sum(bf16_args))


def mul_bf16(*kargs):
    mult = 1
    for i, arg in enumerate(kargs):
        if i == 0:
            mult = truncate_bf16(arg)
        else:
            mult = mult * truncate_bf16(arg)
    return truncate_grad_bf16(mult)


def sigmoid_bf16(x):
    return truncate_grad_bf16(torch.sigmoid(truncate_bf16(x)))


def tanh_bf16(x):
    return truncate_grad_bf16(torch.tanh(truncate_bf16(x)))


def convert_bf16(input):
    output = truncate_bf16(input)
    return truncate_grad_bf16(output)
