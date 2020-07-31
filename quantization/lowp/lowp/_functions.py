from torch.autograd import Function
from lowp._C import truncate_bf16, truncate_fp8, fp32_to_fp8, fp8_to_fp32, quantemu

USE_QUEMU_FOR_BF16=False

# For BF16: roundingMode=0 --> truncate, 1 --> RHAZ, 2 --> stochastic
# QuantEMU: "BFLOAT16_RNE", "BFLOAT16_RHAZ", "BFLOAT16_RTZ", "BFLOAT16_STOCHASTIC"
class TruncateBF16(Function):

    @classmethod
    def forward(cls, ctx, input, inplace, roundingMode):
        if USE_QUEMU_FOR_BF16:
            rMode = ""
            if (roundingMode == 0):
                rMode = "BFLOAT16_RTZ"
            elif (roundingMode == 1):
                rMode = "BFLOAT16_RHAZ"
            elif (roundingMode == 2):
                rMode = "BFLOAT16_STOCHASTIC"
            elif (roundingMode == 3):
                rMode = "BFLOAT16_RNE"
            else:
                raise ValueError("Unsupported rounding mode")
            return quantemu(input, rMode, inplace)
        else:
            return truncate_bf16(input, inplace, roundingMode)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None


class TruncateGradBF16(Function):

    @classmethod
    def forward(cls, ctx, input, roundingMode, debugStr = None):
        ctx.roundingMode = roundingMode
        ctx.debugStr = debugStr
        # We perform clone operation here to ensure that pytorch won't recognize the operation as identity and remove it as part of its optimizations.
        # The alternative could be to mark the tensor dirty explicitly.
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if (ctx.debugStr is not None):
            print(ctx.debugStr)
        if USE_QUEMU_FOR_BF16:
            rMode = ""
            if (ctx.roundingMode == 0):
                rMode = "BFLOAT16_RTZ"
            elif (ctx.roundingMode == 1):
                rMode = "BFLOAT16_RHAZ"
            elif (ctx.roundingMode == 2):
                rMode = "BFLOAT16_STOCHASTIC"
            elif (ctx.roundingMode == 3):
                rMode = "BFLOAT16_RNE"
            else:
                raise ValueError("Unsupported rounding mode")
            grad_input = quantemu(grad_output.contiguous(), rMode, False)
            if (ctx.debugStr is not None):
                print(grad_input)
        else:
            grad_input = truncate_bf16(
                grad_output.contiguous(), False, ctx.roundingMode)
        return grad_input, None, None


class TruncateFP8(Function):

    @classmethod
    def forward(cls, ctx, input, inplace, exp_width, man_width, exp_bias, roundingMode, lfsrVal):
        return truncate_fp8(input, inplace, exp_width, man_width, exp_bias, roundingMode, lfsrVal)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class TruncateGradFP8(Function):

    @classmethod
    def forward(cls, ctx, input, exp_width, man_width, exp_bias, roundingMode, lfsrVal):
        ctx.roundingMode = roundingMode
        ctx.exp_width = exp_width
        ctx.exp_bias = exp_bias
        ctx.man_width = man_width
        ctx.lfsrVal = lfsrVal
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = truncate_fp8(grad_output, False, ctx.exp_width,
                                  ctx.man_width, ctx.exp_bias, ctx.roundingMode, ctx.lfsrVal)
        return grad_input, None, None, None, None, None



class QuantEmu(Function):

    @classmethod
    def forward(cls, ctx, input, mode, inplace):
        return quantemu(input, mode, inplace)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None