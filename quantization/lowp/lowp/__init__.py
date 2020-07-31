import types
import torch
from lowp.functional import truncate_fp8, truncate_bf16,\
    truncate_grad_fp8, truncate_grad_bf16
from functools import partial
import warnings


def _is_fn(object):
    return isinstance(object, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType))
    # return callable(object)


def _dir_fn(module):
    return [func for func in dir(module) if _is_fn(getattr(module, func))]


_ALL_TORCH_FN = set(_dir_fn(torch) + _dir_fn(torch.Tensor))
_AVOID_WRAP = set([
    'is_tensor',
    '__class__',
    '__hash__',
    'clone',
    '__getattribute__',
    'randperm',
    '__new__',
    '__init_subclass__',
    '_make_subclass',
    'is_floating_point',
    'empty',
    '_has_compatible_shallow_copy_type',
    '_list_with_default',
    'set_num_threads',
    'manual_seed',
    'zeros',
    'ones',
    'shape',
    'numel',
    'size',
    'is_grad_enabled',
    'load',
    'save',
    'zero_',
    'flatten',
    'view',
    't',
    'transpose',
    'permute',
    'detach',
    'tensor',
    'from_numpy',
    'as_tensor',
    'get_device'
])
_ALL_TORCH_FN = _ALL_TORCH_FN - _AVOID_WRAP
_ALL_TORCH_FUNCTIONAL_FN = set(_dir_fn(torch)) - _AVOID_WRAP
_PATCH_TORCH_FN = set([
    'pow',
    'mul',
    'mm',
    'dot',
    'bmm',
    'add',
    'sqrt',
    'rsqrt',
    'div',
    'addmm',
    'norm',
    'matmul',
    # 'batch_norm',
    '__pow__',
    '__add__',
    '__div__',
    '__mul__',
    '__iadd__',
    '__idiv__',
    '__imul__',
    '__ifloordiv__',
    '__mod__',
    '__truediv__',
    '__floordiv'
]) - _AVOID_WRAP


_PATCH_TORCH_FUNCTIONAL_FN = set([
    'conv2d',
    'conv1d',
    # 'linear', #patched admm instead
    'adaptive_avg_pool2d',
    'max_pool2d',
    'avg_pool2d',
    'batch_norm'
]) - _AVOID_WRAP


def _wrap_conv2d(input, weight, bias,
                 stride, padding, dilation, groups):
    return dict(input=input, weight=weight), \
        dict(bias=bias, stride=stride, padding=padding,
             dilation=dilation, groups=groups)


def _wrap_linear(input, weight, bias):
    return dict(input=input, weight=weight), \
        dict(bias=bias)


def _wrap_addmm(input, mat1, mat2, alpha=1, beta=1, out=None):
    return dict(mat1=mat1, mat2=mat2), \
        dict(beta=beta, input=input, alpha=alpha, out=out)


def _wrap_batch_norm(input, running_mean, running_var,
                     weight, bias, training, momentum, eps):
    return dict(input=input), \
        dict(running_mean=running_mean, running_var=running_var,
             weight=weight, bias=bias,
             training=training, momentum=momentum, eps=eps)


_WRAP_SPECIAL = {
    'conv2d': _wrap_conv2d,
    'linear': _wrap_linear,
    'batch_norm': _wrap_batch_norm,
    'addmm': _wrap_addmm
}


def default_config(mode):
    if mode == 'BF16':
        return {'input': [truncate_bf16, {}],
                'output': [truncate_grad_bf16, {}]}
    elif mode == 'FP8':
        return {'input': [truncate_fp8, dict(exp_width=5, man_width=2,
                                             exp_bias=15, roundingMode=0,
                                             lfsrVal=0)],
                'output': [truncate_grad_fp8, dict(exp_width=5, man_width=2,
                                                   exp_bias=15, roundingMode=0,
                                                   lfsrVal=0)]}
    elif mode == 'FP8(143)':
        return {'input': [truncate_fp8, dict(exp_width=4, man_width=3,
                                             exp_bias=11, roundingMode=0,
                                             lfsrVal=0)],
                'output': [truncate_grad_fp8, dict(exp_width=5, man_width=2,
                                                   exp_bias=15, roundingMode=0,
                                                   lfsrVal=0)]}
    else:
        raise NotImplementedError


def recursive_wrap(inputs, wrapFN):
    if torch.is_tensor(inputs):
        if inputs.is_cuda and inputs.dtype in [torch.float, torch.half]:
            return wrapFN(inputs)
        else:
            return inputs
    elif isinstance(inputs, list) or isinstance(inputs, tuple):
        out = [recursive_wrap(inp, wrapFN) for inp in inputs]
        if isinstance(inputs, tuple):
            out = tuple(out)
        return out
    elif isinstance(inputs, dict):
        return {recursive_wrap(key, wrapFN): recursive_wrap(val, wrapFN)
                for key, val in inputs.items()}
    else:
        return inputs


def patch_fn(fn, qInputFn, qGradOutputFn, warn=False):
    def wrap_fn(*kargs, **kwargs):

        if fn.__name__ in _WRAP_SPECIAL.keys():
            qkwargs, kwargs = _WRAP_SPECIAL[fn.__name__](*kargs, **kwargs)
            qkwargs = recursive_wrap(qkwargs, qInputFn)
            kwargs.update(qkwargs)
            kargs = []
        else:
            kargs, kwargs = recursive_wrap((kargs, kwargs), qInputFn)
        if warn:
            warnings.warn(
                "<LOWP> function {fname} was patched with lowp on {numargs} args"
                    .format(fname=fn.__name__, numargs=len(kargs)+len(kwargs)),
                stacklevel=2)
        outputs = fn(*kargs, **kwargs)
        fn_output = fn.__name__ + '_output'
        if fn_output in _WRAP_SPECIAL.keys():
            return _WRAP_SPECIAL[fn_output](outputs)
        else:
            return recursive_wrap(outputs, qGradOutputFn)
    return wrap_fn


def warn_fn(fn, name):
    def wrap_fn(*kargs, **kwargs):
        warnings.warn(
            "<LOWP> function %s was used but not patched with lowp" % name,
            stacklevel=2)
        return fn(*kargs, **kwargs)
    return wrap_fn


def remove(module, func_set):
    for fn_name in func_set:
        fn = getattr(module, 'lowp_patched_' + fn_name, None)
        if fn is not None:
            setattr(module, fn_name, fn)
            delattr(module, 'lowp_patched_' + fn_name)
    return True


def patch_module(module, func_set, mode, warn=False):
    if mode == 'None':
        return remove(module, func_set)
    for fn_name in func_set:
        fn = getattr(module, 'lowp_patched_' + fn_name, None)
        if fn is None:  # not patched yet
            fn = getattr(module, fn_name, None)
            if fn is None:
                continue
            setattr(module, 'lowp_patched_' + fn_name, fn)
        defaults = default_config(mode)
        qInputFn, defaults_input = defaults['input']
        qGradOutputFn, defaults_grad = defaults['output']
        qInputFn = partial(qInputFn, **defaults_input)
        qGradOutputFn = partial(qGradOutputFn, **defaults_grad)
        setattr(module, fn_name, patch_fn(
            fn, qInputFn, qGradOutputFn, warn=warn))
    return True


def patch_module_warning(module, func_set, mode):
    if mode == 'None':
        return remove(module, func_set)
    for fn_name in func_set:
        fn = getattr(module, 'lowp_patched_' + fn_name, None)
        if fn is None:  # not patched yet
            fn = getattr(module, fn_name, None)
            if fn is None:
                continue
            setattr(module, 'lowp_patched_' + fn_name, fn)
        setattr(module, fn_name, warn_fn(fn, fn_name))
    return True


def enable(mode='BF16',
           patched_torch=_PATCH_TORCH_FN,
           patched_nn=_PATCH_TORCH_FUNCTIONAL_FN,
           warn_patched=False,
           warn_not_patched=True):
    assert mode in ['BF16', 'FP8', 'FP8(143)', 'None']
    out = True

    unwrapped_torch_fn = _ALL_TORCH_FN - patched_torch
    unwrapped_functional = _ALL_TORCH_FUNCTIONAL_FN - \
        patched_nn

    out &= patch_module(torch, patched_torch, mode, warn=warn_patched)
    out &= patch_module(torch.Tensor, patched_torch, mode, warn=warn_patched)
    out &= patch_module(torch.nn.functional, patched_nn,
                        mode, warn=warn_patched)
    if warn_not_patched:
        out &= patch_module_warning(torch, unwrapped_torch_fn, mode)
        out &= patch_module_warning(torch.Tensor, unwrapped_torch_fn, mode)
        out &= patch_module_warning(
            torch.nn.functional, unwrapped_functional, mode)
    return out


def disable(patched_torch=_PATCH_TORCH_FN,
            patched_nn=_PATCH_TORCH_FUNCTIONAL_FN):
    return enable(mode='None', patched_torch=patched_torch,
                  patched_nn=patched_nn)


class Lowp():
    def __init__(self, mode='BF16', warn_patched=False, warn_not_patched=False,
                 patched_torch=_PATCH_TORCH_FN,
                 patched_nn=_PATCH_TORCH_FUNCTIONAL_FN,
                 exclude=[]):
        if not isinstance(exclude, list):
            exclude = [exclude]

        def _check(item, exclusion_list=exclude):
            if len(exclusion_list) == 0:
                return True
            return all([exc not in item for exc in exclusion_list])
        patched_torch = set([p for p in patched_torch if _check(p)])
        patched_nn = set([p for p in patched_nn if _check(p)])

        self.mode = mode
        self.patched_torch = patched_torch
        self.patched_nn = patched_nn
        self.warn_patched = warn_patched
        self.warn_not_patched = warn_not_patched

    def __enter__(self):
        enable(self.mode, patched_torch=self.patched_torch,
               patched_nn=self.patched_nn,
               warn_patched=self.warn_patched,
               warn_not_patched=self.warn_not_patched)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        disable(patched_torch=self.patched_torch,
                patched_nn=self.patched_nn)
