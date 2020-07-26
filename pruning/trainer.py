import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.utils import clip_grad_norm_
from utils.meters import AverageMeter, accuracy
from utils.mixup import MixUp, CutMix
from random import sample
from models.modules.prunning import ZeroBN,Conv2dStats
import numpy as np
from scipy.optimize import root_scalar


try:
    import tensorwatch

    _TENSORWATCH_AVAILABLE = True
except ImportError:
    _TENSORWATCH_AVAILABLE = False




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


class Trainer(object):


    def saveStats(self, module, grad_input, grad_output):

        grad_output = grad_output[0].contiguous().view(-1)
        module.gradOutputSparsity = torch.numel(( grad_output == 0).nonzero())

        module.gradOutputTau = 0
        module.gradOutputMinusTau = 0
        if torch.numel(torch.unique(grad_output.abs())) > 1:
            v,i = torch.topk(torch.unique(grad_output.abs()),2,largest = False)
            module.gradOutputTau = torch.numel((grad_output == v[1]).nonzero())
            module.gradOutputMinusTau = torch.numel((grad_output == (v[1] * -1)).nonzero())

        module.elems = torch.numel(grad_output)


    def collectGradLayerByLayer(self, module, grad_input, grad_output):

        tens = (torch.abs(grad_input[0][grad_output[0] != 0]))
        tens = torch.log(tens[tens != 0 ]).view(-1).detach()

        mn = torch.mean(tens)
        sd = torch.std(tens)

        sprs = torch.numel((grad_output[0].contiguous().view(-1) ==0).nonzero()) / torch.numel(grad_output[0])
        # if module.fullName == "layer3.1.bn2":
        #     self.sparsified_upper_blocks = 0
        #
        # if sprs  > self.prunRatio:
        #     v = torch.min(grad_output[0].view(-1).abs()[grad_output[0].view(-1) != 0])
        #     idx = (grad_output[0].view(-1).abs_() == v).nonzero()
        #     if torch.numel(idx > 0):
        #         idx = idx[0]
        #
        #     module.final_tau = grad_input[0].view(-1)[idx].abs()
        # else:
        #     right_mode_sparsity = (self.prunRatio - sprs) / (1 - sprs)
        #     guess = torch.tensor([1], device="cuda", dtype=torch.float)
        #     bracket = [torch.exp(torch.tensor([-9], device="cuda", dtype=torch.float)),
        #                torch.exp(torch.tensor([9], device="cuda", dtype=torch.float))]
        #
        #
        #     sol = root_scalar(self.equationStochastic, x0=guess, bracket=bracket, args=(right_mode_sparsity, sd))
        #     threshold = torch.log(torch.tensor([sol.root], device="cuda", dtype=torch.float))
        #
        #     module.final_tau = torch.exp(threshold + mn).detach().cuda()
        #


        # calculate the sparsity required so the overall sparsity will be prunRatio, according to the layers already sparsified
        revised_sparsity = (
                                       self.prunRatio * self.total_numel - self.sparsified_upper_blocks) / self.total_not_preserve_cs
        if (not module.preserve_cosine) and revised_sparsity > module.max_sparsity:
            revised_sparsity = module.max_sparsity

        if ((not module.preserve_cosine) and sprs > revised_sparsity) or (
                module.preserve_cosine and sprs > self.prunRatio):

            v = torch.min(grad_output[0].view(-1).abs()[grad_output[0].view(-1) != 0])
            idx = (grad_output[0].view(-1).abs_() == v).nonzero()
            if torch.numel(idx > 0):
                idx = idx[0]
            module.final_tau = grad_input[0].view(-1)[idx].abs()
        else:

            right_mode_sparsity_prunRatio = (self.prunRatio - sprs) / (1 - sprs)
            right_mode_sparsity_revised_sparsity = (revised_sparsity - sprs) / (1 - sprs)

            guess = torch.tensor([1], device="cuda", dtype=torch.float)
            bracket = [torch.exp(torch.tensor([-9], device="cuda", dtype=torch.float)),
                       torch.exp(torch.tensor([20], device="cuda", dtype=torch.float))]
            k = torch.topk(tens, int(tens.numel() / 300))[0][-1]  # k = percentile 99.66%

            k = (k - mn) / sd
            module.k = k.item()
            module.sigma = sd.item()

            if module.preserve_cosine:
                #                     sol = root_scalar(self.equation_threshold_from_cosine, x0=guess, bracket=bracket, args=(self.cos_sim, sd))

                sol = root_scalar(self.equationStochastic, x0=guess, bracket=bracket,
                                  args=(right_mode_sparsity_prunRatio, sd))
                if self.equation_threshold_from_cosine_truncated(sol.root, 0, sd, k) < module.cos_sim:
                    sol = root_scalar(self.equation_threshold_from_cosine_truncated, x0=guess, bracket=bracket,
                                      args=(module.cos_sim, sd, k))

            else:  # without preserve_cosine
                sol = root_scalar(self.equationStochastic, x0=guess, bracket=bracket,
                                  args=(right_mode_sparsity_revised_sparsity, sd))


            threshold = torch.log(torch.tensor([sol.root], device="cuda", dtype=torch.float))

            module.final_tau = torch.exp(threshold + mn).detach().cuda()




    def __init__(self, model, criterion, optimizer=None,
                 device_ids=[0], device=torch.cuda, dtype=torch.float,
                 distributed=False, local_rank=-1, adapt_grad_norm=None,
                 mixup=None, cutmix=None, loss_scale=1., grad_clip=-1, print_freq=100):
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

            loss.backward()  # accumulate gradient
        grad = clip_grad_norm_(self.model.parameters(), float('inf'))
        return grad

    def _step(self, inputs_batch, target_batch, training=False, average_output=False, chunk_batch=1,ml_logger = None,collectStats = False,first_batch = False):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.training_steps)


        for i, (inputs, target) in enumerate(zip(inputs_batch.chunk(chunk_batch, dim=0),
                                                 target_batch.chunk(chunk_batch, dim=0))):
            target = target.to(self.device)
            inputs = inputs.to(self.device, dtype=self.dtype)

            mixup = None
            if training:
                self.optimizer.pre_forward()

            # compute output
            output = self.model(inputs)

            loss = self.criterion(output, target)


            if training and ml_logger is not None:
                ml_logger.log_metric('Training Loss', loss.item(), step='auto',
                                     log_to_tfboard=False)
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
                    loss = loss * self.loss_scale


                loss.backward()  # accumulate gradient



        if training:  # post gradient accumulation
            if self.loss_scale is not None:
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.data.div_(self.loss_scale)

            if self.grad_clip > 0:
                grad = clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()  # SGD step
            self.training_steps += 1

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss, grad


    def forward(self, data_loader, num_steps=None, training=False, average_output=False, chunk_batch=1, ml_logger = None,collectStats = False,lbl = False):

        meters = {name: AverageMeter()
                  for name in ['step', 'data', 'loss', 'prec1', 'prec5']}
        if training and self.grad_clip > 0:
            meters['grad'] = AverageMeter()

        batch_first = True
        if training and isinstance(self.model, nn.DataParallel) or chunk_batch > 1:
            batch_first = False

        def meter_results(meters):
            results = {name: meter.avg for name, meter in meters.items()}
            results['error1'] = 100. - results['prec1']
            results['error5'] = 100. - results['prec5']
            return results

        end = time.time()
        stepsCollectStats = np.random.permutation(len(data_loader))[:9]
        np.append(stepsCollectStats,0)

        for i, (inputs, target) in enumerate(data_loader):

            # measure data loading time
            meters['data'].update(time.time() - end)

            if collectStats:
                handle = []
                for m in self._model.modules():
                    if isinstance(m,ZeroBN):
                        handle.append(m.register_backward_hook(self.collectGradLayerByLayer))

            if not collectStats and training and ml_logger is not None and i in stepsCollectStats:
                handle2 = []
                for m in self._model.modules():
                    if isinstance(m, Conv2dStats):
                        handle2.append(m.register_backward_hook(self.saveStats))


            output, loss, grad = self._step(inputs, target,
                                            training=training,
                                            average_output=average_output,
                                            chunk_batch=chunk_batch,ml_logger = ml_logger,collectStats = i in stepsCollectStats, first_batch = i==0)


            if collectStats:
                for h in handle:
                    h.remove()

            if not collectStats and training and ml_logger is not None and i in stepsCollectStats:
                for h in handle2:
                    h.remove()


            if training and ml_logger is not None and i in stepsCollectStats:

                totalZeros = 0
                totalMinusTau  = 0
                totalTau = 0
                totalElems = 0
                for m in self.model.modules():
                    if isinstance(m,Conv2dStats):
                        ml_logger.log_metric(m.fullName + 'Grad output sparsifty' , m.gradOutputSparsity / m.elems, step='auto', log_to_tfboard=False)
                        ml_logger.log_metric(m.fullName + 'Grad output Tau' , m.gradOutputTau / m.elems, step='auto', log_to_tfboard=False)
                        ml_logger.log_metric(m.fullName + 'Grad output Minus Tau' , m.gradOutputMinusTau / m.elems, step='auto', log_to_tfboard=False)

                        totalElems += m.elems
                        totalZeros += m.gradOutputSparsity
                        totalMinusTau += m.gradOutputMinusTau
                        totalTau += m.gradOutputTau

                if totalElems > 0:
                    ml_logger.log_metric('Total Zeros', totalZeros / totalElems, step='auto',
                                         log_to_tfboard=False)
                    ml_logger.log_metric('Total Tau', totalTau / totalElems , step='auto',
                                         log_to_tfboard=False)
                    ml_logger.log_metric('Total Minus Tau', totalMinusTau / totalElems, step='auto',
                                         log_to_tfboard=False)



            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            meters['loss'].update(float(loss), inputs.size(0))
            meters['prec1'].update(float(prec1), inputs.size(0))
            meters['prec5'].update(float(prec5), inputs.size(0))
            if grad is not None:
                meters['grad'].update(float(grad), inputs.size(0))



            # measure elapsed time
            meters['step'].update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 or i == len(data_loader) - 1:
                if training and ml_logger is not None:
                    ml_logger.log_metric('Train Acc1', meters['prec1'].avg, step='auto', log_to_tfboard=False)
                    ml_logger.log_metric('Train Acc5', meters['prec5'].avg, step='auto', log_to_tfboard=False)
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
                    report += 'Grad {meters[grad].val:.3f} ({meters[grad].avg:.3f})' \
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


        return meter_results(meters)



    def train(self, data_loader, ml_logger=None,average_output=False, chunk_batch=1,num_steps = None):
        # switch to train mode
        self.model.train()
        self.write_stream('epoch', (self.training_steps, self.epoch))
        return self.forward(data_loader, training=True, average_output=average_output, chunk_batch=chunk_batch, ml_logger = ml_logger,num_steps = num_steps)

    def validate(self, data_loader, average_output=False):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            return self.forward(data_loader, average_output=average_output, training=False )



    def collectStat(self, data_loader,num_steps,prunRatio = 0, cos_sim=0, cos_sim_max=0):
        self.model.train()

        self.prunRatio = prunRatio
        self.cos_sim = cos_sim
        self.cos_sim_max = cos_sim_max
        # #CIFAR10 -ResNet
        # self.numel_block1 = 2**22
        # self.numel_block2 = 2**21
        # self.numel_block3 = 2**20

        # CIFAR100 -ResNet
        self.numel_block1 = 2 ** 22
        self.numel_block2 = 2 ** 21
        self.numel_block3 = 2 ** 20
        self.layers_per_block = 5

        self.total_numel = self.layers_per_block * (self.numel_block3 + self.numel_block2 + self.numel_block1)
        self.total_not_preserve_cs = self.layers_per_block * self.numel_block1

        # #Imagenet - ResNet
        # self.numel_block0 = 205520896
        # self.numel_block1 = 51380224
        # self.numel_block2 = 25690112
        # self.numel_block3 = 12845056
        # self.numel_block4 = 6422528
        # self.layers_per_block = 5

        # self.total_numel = self.layers_per_block*(self.numel_block4 + self.numel_block3 + self.numel_block2 + self.numel_block1) + self.numel_block0
        # self.total_not_preserve_cs = (self.layers_per_block * self.numel_block1) + self.numel_block0

        # Imagenet - VGG

        # self.numel_block1 = 102760448
        # self.numel_block2 = 51380224
        # self.numel_block3 = 25690112
        # self.numel_block4 = 12845056
        # self.numel_block5 = 3211264
        # self.layers_per_block1_2 = 2
        # self.layers_per_block3_4_5 = 3
        #
        # self.total_numel = self.layers_per_block1_2*(self.numel_block2 + self.numel_block1) + self.layers_per_block3_4_5*(self.numel_block5 + self.numel_block4 + self.numel_block3)
        # self.total_not_preserve_cs = self.layers_per_block1_2*(self.numel_block2 + self.numel_block1)
        # self.total_not_preserve_cs = self.numel_block0

        self.sparsified_upper_blocks = 0
        self.forward(data_loader, num_steps=num_steps, training=True,collectStats = True)



    def equationStochastic(self, alpha, sparsity, sigma):

        sqrt2 = torch.sqrt(torch.tensor([2], device="cuda", dtype=torch.float))
        alpha = torch.tensor([alpha], device="cuda", dtype=torch.float)

        pt1 = torch.exp((sigma**2)/2) * torch.erf(sigma/sqrt2 - torch.log(alpha)/(sqrt2 * sigma))
        pt2 = alpha * torch.erf(torch.log(alpha)/(sqrt2 * sigma))
        pt3 = torch.exp((sigma**2)/2)

        return 0.5 - sparsity + 1/(2*alpha) * (pt1 + pt2 - pt3)

    def equation_threshold_from_cosine_truncated(self, alpha, cos_theta, sigma, k):
        sqrt2 = torch.sqrt(torch.tensor([2], device="cuda", dtype=torch.float))
        alpha = torch.tensor([alpha], device="cuda", dtype=torch.float)

        truncation_arg = torch.erf((k * sigma - 2 * sigma ** 2) / (sqrt2 * sigma))

        numerator = torch.exp(2 * sigma ** 2) * (1 + truncation_arg)
        denom1 = alpha * torch.exp(sigma ** 2 / 2) * (1 - torch.erf((sigma ** 2 - torch.log(alpha)) / (sqrt2 * sigma)))
        denom2 = torch.exp(2 * sigma ** 2) * (
                    truncation_arg - torch.erf((torch.log(alpha) - 2 * sigma ** 2) / (sqrt2 * sigma)))
        denom3 = torch.exp(2 * sigma ** 2) * (1 + truncation_arg)

        return torch.sqrt(numerator / (denom1 + denom2)) - cos_theta

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