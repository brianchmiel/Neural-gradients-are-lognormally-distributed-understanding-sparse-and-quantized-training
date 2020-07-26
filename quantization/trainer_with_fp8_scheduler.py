import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from utils.mixup import MixUp, CutMix
from random import sample
import pdb
import math
import numpy as np
try:
    import tensorwatch
    _TENSORWATCH_AVAILABLE = True
except ImportError:
    _TENSORWATCH_AVAILABLE = False

def find_opt_exp_bits(total_bits, curr_sigma):
    data = {}
    data[3] = [0, 0.7, 10]
    data[4] = [0, 0.5, 1.2, 10]
    data[5] = [0, 0.2, 1, 2.4, 10]
    data[6] = [0, 0.3, 0.9, 2, 4.9, 40]
    data[7] = [0, 0.3, 0.7, 1.7, 4, 9.4, 40]
    data[8] = [0, 0.1, 0.6, 1.4, 3.7, 8.1, 19.3, 40]
    pdb.set_trace()
    curr_data = data[total_bits]
    print("curr_data before rounding: " + str(curr_data-curr_sigma))
    curr_data = math.ceil((curr_data-curr_sigma)*10)/10
    print("curr_data after rounding: " + str(curr_data))

    exp_bits = next(k for k, value in enumerate(curr_data) if value > 0)

    return exp_bits



def _flatten_duplicates(inputs, target, batch_first=True, expand_target=True):
    duplicates = inputs.size(1)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    inputs = inputs.flatten(0, 1)

    if expand_target:
        if batch_first:
            target = target.view(-1, 1).expand(-1, duplicates)
        else:
            target = target.view(1, -1).expand(duplicates, -1)
        target = target.flatten(0, 1)
    return inputs, target


def _average_duplicates(outputs, target, batch_first=True):
    """assumes target is not expanded (target.size(0) == batch_size) """
    batch_size = target.size(0)
    reduce_dim = 1 if batch_first else 0
    if batch_first:
        outputs = outputs.view(batch_size, -1, *outputs.shape[1:])
    else:
        outputs = outputs.view(-1, batch_size, *outputs.shape[1:])
    outputs = outputs.mean(dim=reduce_dim)
    return outputs


def _mixup(mixup_modules, alpha, batch_size):
    mixup_layer = None
    if len(mixup_modules) > 0:
        for m in mixup_modules:
            m.reset()
        mixup_layer = sample(mixup_modules, 1)[0]
        mixup_layer.sample(alpha, batch_size)
    return mixup_layer

class Hook:
    def __init__(self, module, name, backward=False):
        self.backward_flag = backward
        self.name = name
        if backward == False:
            # if "relu" in name:
            #     self.hook = module.register_forward_hook(self.hook_fn_relu)
            #     # self.hook = module.register_forward_hook(self.snr_max_calc)
            # else:
            #     self.hook = module.register_forward_hook(self.hook_fn)
            # self.hook = module.register_forward_hook(self.hook_fn_relu)
            pass
        else:
            # self.hook = module.register_backward_hook(self.hook_fn)
            self.hook = module.register_backward_hook(self.collect_stats_and_update_fp_grad_args)

    #
    #
    # def hook_fn(self, module, input, output):
    #     # print("module name: " + str(self.name) + "; BK: " + str(self.backward_flag))
    #     # if "fc" in self.name:
    #     #     pdb.set_trace()
    #     if isinstance(input[0], torch.Tensor):
    #         t_in = input[0].clone().cpu().detach().numpy()
    #     elif isinstance(input[1], torch.Tensor):
    #         t_in = input[1].clone().cpu().detach().numpy()
    #     elif isinstance(input, torch.Tensor):
    #         t_in = input.clone().cpu().detach().numpy()
    #
    #     if isinstance(output[0], torch.Tensor):
    #         t_out = output[0].clone().cpu().detach().numpy()
    #     elif isinstance(output[1], torch.Tensor):
    #         t_out = output[1].clone().cpu().detach().numpy()
    #     elif isinstance(output, torch.Tensor):
    #         t_out = output.clone().cpu().detach().numpy()
    #     #
    #     # if input[0] is None:
    #     #     pdb.set_trace()
    #     #     t_in = input.clone().cpu().detach().numpy()
    #     # else:
    #     #     t_in = input[0].clone().cpu().detach().numpy()
    #     # if output[0] is None:
    #     #     t_out = output.clone().cpu().detach().numpy().copy()
    #     # else:
    #     #     t_out = output[0].clone().cpu().detach().numpy().copy()
    #     # print("in_mean_res: " + str(np.mean(abs(t_in))) + "; out_mean_res: " + str(np.mean(abs(t_out))))
    #     in_std_res = np.std(np.log2(np.abs(t_in)))
    #     # out_std_res = np.std(np.log2(np.abs(t_out)))
    #     in_mean_res = np.mean(np.log2(np.abs(t_in)))
    #     # out_mean_res = np.mean(np.log2(np.abs(t_out)))
    #     # print("in_mean_res: " + str(in_mean_res) + "; in_std_res: " + str(in_std_res))
    #     # print("out_mean_res: " + str(out_mean_res) + "; out_std_res: " + str(out_std_res))
    #     # print("in_mean_res: " + str(in_mean_res) + "; out_mean_res: " + str(out_mean_res))
    #     # print("in_std_res: " + str(in_std_res) + "; out_std_res: " + str(out_std_res))
    #     # print("module name: " + str(self.name))
    #     self.grad_log_mean = in_mean_res
    #     self.grad_log_std = in_std_res
    #     # self.grad_log_mean = 0
    #     # self.grad_log_std = 0
    def hook_fn(self, module, input, output):
        # print("module name: " + str(self.name) + "; BK: " + str(self.backward_flag))
        # if "fc" in self.name:
        #     pdb.set_trace()
        # if isinstance(input[0], torch.Tensor):
        #     t_in = input[0].clone().detach()
        # elif isinstance(input[1], torch.Tensor):
        #     t_in = input[1].clone().detach()
        # elif isinstance(input, torch.Tensor):
        #     t_in = input.clone().detach()
        # pdb.set_trace()
        if isinstance(output[0], torch.Tensor):
            t_out = output[0].clone().detach()
        elif isinstance(output[1], torch.Tensor):
            t_out = output[1].clone().detach()
        elif isinstance(output, torch.Tensor):
            t_out = output.clone().detach()
        # t_out_ = t_out.reshape(-1)
        # t_out_ = t_out_[t_out_.nonzero().reshape(-1)]
        #
        # if input[0] is None:
        #     pdb.set_trace()
        #     t_in = input.clone().cpu().detach().numpy()
        # else:
        #     t_in = input[0].clone().cpu().detach().numpy()
        # if output[0] is None:
        #     t_out = output.clone().cpu().detach().numpy().copy()
        # else:
        #     t_out = output[0].clone().cpu().detach().numpy().copy()
        # print("in_mean_res: " + str(np.mean(abs(t_in))) + "; out_mean_res: " + str(np.mean(abs(t_out))))
        # pdb.set_trace()
        # log2_tensor = torch.log2(torch.abs(t_in))
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # # pdb.set_trace()
        # in_std_res = torch.std(log2_tensor)
        # # in_mean_res = torch.mean(log2_tensor)
        # t_in[t_in == float("inf")] = 57344
        # t_in[t_in == float("-inf")] = -57344
        # log2_tensor_in = torch.log2(torch.abs(t_in))
        # print("min" + str(log2_tensor.min()))
        # print("max" + str(log2_tensor.max()))
        # if log2_tensor == float("nan"):
        #     pdb.set_trace()
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # log2_tensor_in[log2_tensor_in == float("-inf")] = -14
        # log2_tensor_in[log2_tensor_in == float("inf")] = 14
        # pdb.set_trace()
        # in_std_res = torch.std(log2_tensor_in)
        # in_mean_res = torch.mean(log2_tensor_in)

        # t_out[t_out == float("inf")] = 57344
        # t_out[t_out == float("-inf")] = -57344

        # t_out = t_in
        log2_tensor_out = torch.log2(torch.abs(t_out))
        # print("min" + str(log2_tensor.min()))
        # print("max" + str(log2_tensor.max()))
        # if log2_tensor == float("nan"):
        #     pdb.set_trace()
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # print("INSIDE BKW HOOK")
        log2_tensor_out[log2_tensor_out == float("-inf")] = -7
        log2_tensor_out[log2_tensor_out == float("inf")] = 7

        out_std_res = torch.std(log2_tensor_out)
        out_mean_res = torch.mean(log2_tensor_out)
        # if self.name == 'layer3.1.conv2':
        # if math.isnan(out_std_res):
        #     pdb.set_trace()
        # out_std_res = np.std(np.log2(np.abs(t_out)))
        # out_mean_res = np.mean(np.log2(np.abs(t_out)))
        # print("in_mean_res: " + str(in_mean_res) + "; in_std_res: " + str(in_std_res))
        # print("inside hook module name: " + str(self.name) + " mean_grad: " + str(out_mean_res) + " std_grad: " + str(out_std_res))
        # pdb.set_trace()
        # print("out_mean_res: " + str(out_mean_res) + "; out_std_res: " + str(out_std_res))
        # print("in_mean_res: " + str(in_mean_res) + "; out_mean_res: " + str(out_mean_res))
        # print("in_std_res: " + str(in_std_res) + "; out_std_res: " + str(out_std_res))
        # print("module name: " + str(self.name))
        # self.grad_log_mean = in_mean_res
        # self.grad_log_std = in_std_res
        self.grad_log_mean = out_mean_res
        self.grad_log_std = out_std_res
        # self.grad_log_mean = 0
        # self.grad_log_std = 0
        # pdb.set_trace()

    def collect_stats_and_update_fp_grad_args(self, module, input, output):
        # print("module name: " + str(self.name) + "; BK: " + str(self.backward_flag))
        # if "fc" in self.name:
        #     pdb.set_trace()
        # if isinstance(input[0], torch.Tensor):
        #     t_in = input[0].clone().detach()
        # elif isinstance(input[1], torch.Tensor):
        #     t_in = input[1].clone().detach()
        # elif isinstance(input, torch.Tensor):
        #     t_in = input.clone().detach()
        # pdb.set_trace()
        if isinstance(output[0], torch.Tensor):
            t_out = output[0].clone().detach()
        elif isinstance(output[1], torch.Tensor):
            t_out = output[1].clone().detach()
        elif isinstance(output, torch.Tensor):
            t_out = output.clone().detach()
        # t_out_ = t_out.reshape(-1)
        # t_out_ = t_out_[t_out_.nonzero().reshape(-1)]
        #
        # if input[0] is None:
        #     pdb.set_trace()
        #     t_in = input.clone().cpu().detach().numpy()
        # else:
        #     t_in = input[0].clone().cpu().detach().numpy()
        # if output[0] is None:
        #     t_out = output.clone().cpu().detach().numpy().copy()
        # else:
        #     t_out = output[0].clone().cpu().detach().numpy().copy()
        # print("in_mean_res: " + str(np.mean(abs(t_in))) + "; out_mean_res: " + str(np.mean(abs(t_out))))
        # pdb.set_trace()
        # log2_tensor = torch.log2(torch.abs(t_in))
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # # pdb.set_trace()
        # in_std_res = torch.std(log2_tensor)
        # # in_mean_res = torch.mean(log2_tensor)
        # t_in[t_in == float("inf")] = 57344
        # t_in[t_in == float("-inf")] = -57344
        # log2_tensor_in = torch.log2(torch.abs(t_in))
        # print("min" + str(log2_tensor.min()))
        # print("max" + str(log2_tensor.max()))
        # if log2_tensor == float("nan"):
        #     pdb.set_trace()
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # log2_tensor_in[log2_tensor_in == float("-inf")] = -14
        # log2_tensor_in[log2_tensor_in == float("inf")] = 14
        # pdb.set_trace()
        # in_std_res = torch.std(log2_tensor_in)
        # in_mean_res = torch.mean(log2_tensor_in)

        # t_out[t_out == float("inf")] = 57344
        # t_out[t_out == float("-inf")] = -57344


        log2_tensor_out = torch.log2(torch.abs(t_out))
        # print("min" + str(log2_tensor.min()))
        # print("max" + str(log2_tensor.max()))
        # if log2_tensor == float("nan"):
        #     pdb.set_trace()
        # log2_tensor[log2_tensor == float("-inf")] = -6
        # print("INSIDE BKW HOOK")
        log2_tensor_out[log2_tensor_out == float("-inf")] = -126
        log2_tensor_out[log2_tensor_out == float("inf")] = 126

        out_std_res = torch.std(log2_tensor_out)
        out_mean_res = torch.mean(log2_tensor_out)

        module.mu = out_mean_res
        module.sigma = out_std_res
        total_bits_num = module.fp_x
        module.loss_scale.fill_(2**(-out_mean_res))  # update loss scale

        exp_bits = find_opt_exp_bits(total_bits=total_bits_num, curr_sigma=out_std_res)
        module.fp8_grad_args = dict(exp_width=exp_bits, man_width=(total_bits_num-1-exp_bits), exp_bias=(2**(exp_bits-1))-1, roundingMode=0, lfsrVal=0)

        # if self.name == 'layer3.1.conv2':
        # if math.isnan(out_std_res):
        #     pdb.set_trace()
        # out_std_res = np.std(np.log2(np.abs(t_out)))
        # out_mean_res = np.mean(np.log2(np.abs(t_out)))
        # print("in_mean_res: " + str(in_mean_res) + "; in_std_res: " + str(in_std_res))
        # print("inside hook module name: " + str(self.name) + " mean_grad: " + str(out_mean_res) + " std_grad: " + str(out_std_res))
        # pdb.set_trace()
        # print("out_mean_res: " + str(out_mean_res) + "; out_std_res: " + str(out_std_res))
        # print("in_mean_res: " + str(in_mean_res) + "; out_mean_res: " + str(out_mean_res))
        # print("in_std_res: " + str(in_std_res) + "; out_std_res: " + str(out_std_res))
        # print("module name: " + str(self.name))
        # self.grad_log_mean = in_mean_res
        # self.grad_log_std = in_std_res
        self.grad_log_mean = out_mean_res
        self.grad_log_std = out_std_res
        # self.grad_log_mean = 0
        # self.grad_log_std = 0
        # pdb.set_trace()

    def close(self):
        self.hook.remove()


def fine_module_by_name(model, name_in):
    for name, module in model.named_modules():
        if name == name_in:
            return module

class Trainer(object):

    def __init__(self, model, criterion, optimizer=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None,
                 mixup=None, cutmix=None, loss_scale=1., grad_clip=-1, print_freq=100, enable_input_grad_statistics=False):
        self._model = model
        self.criterion = criterion
        self.epoch = 0
        self.training_steps = 0
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype
        self.distributed = distributed
        self.local_rank = local_rank
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        self.mixup = mixup
        self.cutmix = cutmix
        self.grad_scale = None
        self.loss_scale = loss_scale
        self.adapt_grad_norm = adapt_grad_norm
        self.watcher = None
        self.streams = {}

        ## moran
        self.enable_input_grad_statistics = enable_input_grad_statistics
        self.input_grad_statistics = None
        self.module_to_hook = {} # TODO:: add to this dict

        if self.enable_input_grad_statistics:
            layer_name = []
            for name, module in self._model.named_modules():
                # pdb.set_trace()
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    layer_name.append(name)
            # pdb.set_trace()
            for name in layer_name:
                self.module_to_hook[name] = fine_module_by_name(self._model, name)


        ## moran- end

        if distributed:
            self.model = nn.parallel.DistributedDataParallel(model,
                                                             device_ids=device_ids,
                                                             output_device=device_ids[0])
        elif device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids)
        else:
            self.model = model

    def _grad_norm(self, inputs_batch, target_batch, chunk_batch=1):
        self.model.zero_grad()
        for inputs, target in zip(inputs_batch.chunk(chunk_batch, dim=0),
                                  target_batch.chunk(chunk_batch, dim=0)):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            # compute output
            output = self.model(inputs)
            loss = self.criterion(output, target)

            if chunk_batch > 1:
                loss = loss / chunk_batch

            loss.backward()   # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1, scheduled_instructions=None):
        outputs = []
        total_loss = 0
        # self.input_grad_statistics = True if ((self.epoch > -1) and self.enable_input_grad_statistics) and (float(self.epoch) % 2 == 0) else False
        # if scheduled_instructions is None:
        #     self.input_grad_statistics = True if ((self.epoch > -1) and self.enable_input_grad_statistics) else False
        # else:
        #     self.input_grad_statistics = scheduled_instructions['collect_stat']

        meters = {name: {'mean': AverageMeter(), 'std': AverageMeter()} for name in self.module_to_hook.keys()}
        grad_log_stats = {}

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)

        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):

            self.input_grad_statistics = True if ((self.epoch > -1) and self.enable_input_grad_statistics) and (float(self.epoch) % 2 == 0) and i==0 else False
            if training:
                if self.input_grad_statistics:
                    BK_hookF = {}
                    for name, module in self.module_to_hook.items():
                        # print("name: " + str(name))
                        BK_hookF[name] = Hook(module, name, True)

            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            mixup = None
            if training:
                self.optimizer.pre_forward()
                if self.mixup is not None or self.cutmix is not None:
                    input_mixup = CutMix() if self.cutmix else MixUp()
                    mix_val = self.mixup or self.cutmix
                    mixup_modules = [input_mixup]  # input mixup
                    mixup_modules += [m for m in self.model.modules()
                                      if isinstance(m, MixUp)]
                    mixup = _mixup(mixup_modules, mix_val, inputs.size(0))
                    inputs = input_mixup(inputs)

            # compute output
            output = self.model(inputs)

            if mixup is not None:
                target = mixup.mix_target(target, output.size(-1))

            if average_output:
                if isinstance(output, list) or isinstance(output, tuple):
                    output = [_average_duplicates(out, target) if out is not None else None
                              for out in output]
                else:
                    output = _average_duplicates(output, target)
            loss = self.criterion(output, target)
            grad = None

            if chunk_batch > 1:
                loss = loss / chunk_batch

            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            outputs.append(output.detach())
            total_loss += float(loss)

            if training:
                if i == 0:
                    self.optimizer.pre_backward()
                if self.grad_scale is not None:
                    loss = loss * self.grad_scale
                if self.loss_scale is not None:
                    pass  # moran
                    # pdb.set_trace()
                    # loss = loss * self.loss_scale
                loss.backward()   # accumulate gradient


                ## moran- gather lognorm statistics


                if self.input_grad_statistics:
                    # pdb.set_trace()
                    # total = reduce((lambda total, item: total+item.alloc), snds)
                    for hook in BK_hookF.values():
                        meters[hook.name]['mean'].update(float(hook.grad_log_mean), inputs.size(0))
                        meters[hook.name]['std'].update(float(hook.grad_log_std), inputs.size(0))
                    # curr_grad_log_stats = {hook.name: {'mean': hook.grad_log_mean, 'std': hook.grad_log_std} for hook in BK_hookF.values()}
                    # pdb.set_trace()

                if self.input_grad_statistics:
                    for hook in BK_hookF.values():
                        hook.close()
                ## moran- end



        grad_log_stats = {name: {'mean': met['mean'].avg, 'std': met['std'].avg} for name, met in meters.items()}
        # print("grad_log_stats! loop 1")
        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    # p.grad.data.div_(self.loss_scale)  # moran

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # dont_update_grad = False
            # for param in self.model.parameters():
            #     param.grad[torch.isnan(param.grad)] = 0
                # if torch.isnan(param.grad).any():  # moran
                #     pdb.set_trace()
                #     dont_update_grad = True
                #     break
            # if dont_update_grad:
            #     pass
            # else:
            #     self.optimizer.step()  # SGD step
            self.optimizer.step()  # SGD step
            self.training_steps += 1

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad, grad_log_stats

    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1, scheduled_instructions=None):

        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        meters_grad = {name: {'mean': AverageMeter(), 'std': AverageMeter()} for name in self.module_to_hook.keys()}

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()

        for i, (inputs, target) in enumerate(data_loader):
            duplicates = inputs.dim() > 4  # B x D x C x H x W
            if training and duplicates and self.adapt_grad_norm is not None \
                    and i % self.adapt_grad_norm == 0:
                grad_mean = 0
                num = inputs.size(1)
                for j in range(num):
                    grad_mean += float(self._grad_norm(inputs.select(1, j), target))
                grad_mean /= num
                grad_all = float(self._grad_norm(
                    *_flatten_duplicates(inputs, target, batch_first)))
                self.grad_scale = grad_mean / grad_all
                logging.info('New loss scale: %s', self.grad_scale)

            # measure data loading time
            meters['data'].update(time.time() - end)
            if duplicates:  # multiple versions for each sample (dim 1)
                inputs, target = _flatten_duplicates(inputs, target, batch_first,
                                                     expand_target=not average_output)

            output, loss, grad, grad_log_stats = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch, scheduled_instructions=scheduled_instructions)

            # print("grad_log_stats!!!")
            # print(grad_log_stats)
            # pdb.set_trace()
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))

            for name, met in meters_grad.items():
                met['mean'].update(float(grad_log_stats[name]['mean']), inputs.size(0))
                met['std'].update(float(grad_log_stats[name]['std']), inputs.size(0))


            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                report = str('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {meters[step].val:.3f} ({meters[step].avg:.3f})\t'
                             'Data {meters[data].val:.3f} ({meters[data].avg:.3f})\t'
                             'Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t'
                             'Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t'
                             'Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t'
                             .format(
                                 self.epoch, i, len(data_loader),
                                 phase='TRAINING' if training else 'EVALUATING',
                                 meters=meters))
                if 'grad' in meters.keys():
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})'\
                        .format(meters=meters)
                logging.info(report)
                self.observe(trainer=self,
                             model=self._model,
                             optimizer=self.optimizer,
                             data=(inputs, target))
                self.stream_meters(meters,
                                   prefix='train' if training else 'eval')
                if training:
                    self.write_stream('lr',
                                      (self.training_steps, self.optimizer.get_lr()[0]))

            if num_steps is not None and i >= num_steps:
                break

        # print("grad_log_stats! loop 2")
        if training:
            for name, met in meters_grad.items():
                print("module name: " + str(name) + " mean_grad: " + str(met['mean'].avg) + " std_grad: " + str(met['std'].avg))

        return meter_results(meters), meters_grad

    def train(self, data_loader, average_output=False, chunk_batch=1, scheduled_instructions=None):
        # switch to train mode
        self.model.train()
        self.write_stream('epoch', (self.training_steps, self.epoch))
        return self.forward(data_loader, training=True, average_output=average_output, chunk_batch=chunk_batch, scheduled_instructions=scheduled_instructions)

    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False)

    def calibrate_bn(self, data_loader, num_steps=None):
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = None
                m.track_running_stats = True
                m.reset_running_stats()
        self.model.train()
        with torch.no_grad():
            return self.forward(data_loader, num_steps=num_steps, training=False)

    ###### tensorwatch methods to enable training-time logging ######

    def set_watcher(self, filename, port=0):
        if not _TENSORWATCH_AVAILABLE:
            return False
        if self.distributed and self.local_rank > 0:
            return False
        self.watcher = tensorwatch.Watcher(filename=filename, port=port)
        # default streams
        self._default_streams()
        self.watcher.make_notebook()
        return True

    def get_stream(self, name, **kwargs):
        if self.watcher is None:
            return None
        if name not in self.streams.keys():
            self.streams[name] = self.watcher.create_stream(name=name,
                                                            **kwargs)
        return self.streams[name]

    def write_stream(self, name, values):
        stream = self.get_stream(name)
        if stream is not None:
            stream.write(values)

    def stream_meters(self, meters_dict, prefix=None):
        if self.watcher is None:
            return False
        for name, value in meters_dict.items():
            if prefix is not None:
                name = '_'.join([prefix, name])
            value = value.val
            stream = self.get_stream(name)
            if stream is None:
                continue
            stream.write((self.training_steps, value))
        return True

    def observe(self, **kwargs):
        if self.watcher is None:
            return False
        self.watcher.observe(**kwargs)
        return True

    def _default_streams(self):
        self.get_stream('train_loss')
        self.get_stream('eval_loss')
        self.get_stream('train_prec1')
        self.get_stream('eval_prec1')
        self.get_stream('lr')
