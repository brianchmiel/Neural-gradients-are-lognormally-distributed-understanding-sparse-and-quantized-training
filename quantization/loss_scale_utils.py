import torch.nn as nn
# from lowp.modules import Conv2d_PerLayer_FP8_BKW
import os
import pandas as pd
import pdb
import numpy as np

# class PerLayerLossScaleSettings:
#     def __init__(self, model, config, args, start_to_collect_stats_in_epoch= 3, collect_stats_every_epochs=None, dynamic_update=False ):
#         self.model = model
#         self.args = args
#         self.args = args
#         self.config = config
#         self.start_to_collect_stats_in_epoch = start_to_collect_stats_in_epoch
#         self.collect_stats_every_epochs = collect_stats_every_epochs
#         self.dynamic_update = dynamic_update
#         # collect stats for loss scale and exp bit -width per layer
#         # add an option to read json/pickle file
#
#
#         # replace nn modules with their
#
#
#
#         layer_name = []
#         for name, module in self.model.named_modules():
#             # pdb.set_trace()
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 layer_name.append(name)
#         # pdb.set_trace()
#         for name in layer_name:
#             self.module_to_hook[name] = fine_module_by_name(self._model, name)
#

# ******
# collect stats
# ******
# 1. collect stats offline- before training begin
# 2. collect stats for one epoch when training starts
# 3. collect stats every x epochs

# ******
# update loss scale and bit-width exp
# ******
# 1. update one time at the beginning from offline stats
# 2. update one time at the beginning after stats collection
# 3. update after every stats collection


class FP8TrainingScheduler:
    def __init__(self, model, config, args, collect_stats_online= False,
                 start_to_collect_stats_in_epoch=3,
                 collect_stats_every_epochs=10,
                 online_update=False,
                 first_update_with_stats_from_epoch=3,
                 start_online_update_in_epoch=None,
                 update_every_epochs=None,
                 update_loss_scale=True,
                 update_exp_bit_width=False,
                 stats_path=None,
                 quantize_modules_name=None,
                 enable_scheduler=True):
        self.enable_scheduler = enable_scheduler
        self.model = model
        self.args = args
        self.args = args
        self.config = config

        # schedule stats collection
        self.collect_stats_online = collect_stats_online  # if offline- read stats from pickle/json, else collect during training
        self.start_to_collect_stats_in_epoch = start_to_collect_stats_in_epoch # if collect_stats_online is offline collect stats from pickle at epoch start_to_collect_stats_in_epoch
        self.collect_stats_every_epochs = collect_stats_every_epochs if self.collect_stats_online else None  # relevant only for online stats collection

        # schedule loss scale and bit-width exp updating
        self.online_update = online_update # if offline- update before epoch 0 with curr update, else: update first before epoch 0 and then update when there is new stats
        self.first_update_with_stats_from_epoch = first_update_with_stats_from_epoch
        if self.online_update:
            self.start_online_update_in_epoch = (self.start_to_collect_stats_in_epoch+1) if start_online_update_in_epoch is None else start_online_update_in_epoch
            self.update_every_epochs = self.collect_stats_every_epochs if update_every_epochs is None else update_every_epochs
        self.update_loss_scale = update_loss_scale
        self.update_exp_bit_width = update_exp_bit_width
        self.quantize_modules_name = quantize_modules_name  # list



        self.stats_collection = None
        self.quantization_wrappers = {}  # dict: {key=name: val=module}

        # stats were collected offline. need to read pickle/json/...
        if not self.collect_stats_online:
            self.stats_collection = self._read_stats_from_csv(stats_path)

        # replace regular modules with fp8 modules
        # self._replace_modules()

        self._create_fp8_modules_list()


        # internal scheduler
        self.scheduled_instructions = {'collect_stat': False}

    def is_enable(self):
        return self.enable_scheduler
    @staticmethod
    def _read_stats_from_csv(stats_path):
        stats_collection = pd.DataFrame()
        if os.path.isfile(stats_path):
            stats_collection = pd.read_csv(stats_path)
        else:
            raise ValueError('{} isn''t a file'.format(stats_path))
        return stats_collection

    def _replace_modules(self):
        for name, m in self.model.named_modules():
            if name in self.quantize_modules_name:
                if isinstance(m, nn.Conv2d):
                    module_wrapper = Conv2d_PerLayer_FP8_BKW(name=name, wrapped_module=m)
                if isinstance(m, nn.Linear):
                    pass
                setattr(self.model, name, module_wrapper)
                self.quantization_wrappers[name] = module_wrapper

    def _create_fp8_modules_list(self):
        for name, m in self.model.named_modules():
            # print("name: " + str(name) + "isintance: " + str(isinstance(m, nn.Conv2d)))
            # if name == 'conv1':
                # pdb.set_trace()
            if name in self.quantize_modules_name:
                if isinstance(m, nn.Conv2d):
                    self.quantization_wrappers[name] = m
        # pdb.set_trace()

    def schedule_before_epoch(self, epoch):
        if not self.is_enable():
            return
        # scheduled_instructions = {}
        # prev_scheduled_instructions  = self.scheduled_instructions.copy()


        # schedule collecting stats in trainer
        if self.collect_stats_online:
            if epoch < self.start_to_collect_stats_in_epoch:
                self.scheduled_instructions['collect_stat'] = False
            else:
                if ((epoch - self.start_to_collect_stats_in_epoch) % self.collect_stats_every_epochs) == 0:
                    self.scheduled_instructions['collect_stat'] = True
                else:
                    self.scheduled_instructions['collect_stat'] = False

        # self.scheduled_instructions = scheduled_instructions


        # update loss scale and exp bit
        if epoch == 0:
            self.update_fp8_training_params(epoch)
        if self.online_update and ((epoch - self.start_online_update_in_epoch) % self.update_every_epochs) == 0 and (epoch - self.start_online_update_in_epoch) >= 0:
            self.update_fp8_training_params(epoch)

    def update_fp8_training_params(self, epoch):
        # need to take into consideration the fields: self.update_loss_scale, update_exp_bit_width
        # pdb.set_trace()
        for name, m in self.quantization_wrappers.items():
            # ...
            if self.update_loss_scale:
                update_from_epoch = self.first_update_with_stats_from_epoch if epoch < self.first_update_with_stats_from_epoch else epoch
                loss_scale_factor = 2**(-self.stats_collection.loc[update_from_epoch]['grad mean ' + name])
                # loss_scale_factor = 2**19
                # loss_scale_factor = 1
                if self.collect_stats_online:
                    # m.loss_scale = m.loss_scale*loss_scale_factor
                    # m.loss_scale.fill_(m.loss_scale*loss_scale_factor)
                    m.loss_scale.fill_(loss_scale_factor)
                    print("0 module name: " + str(name) + "; loss scale: " + str(np.log2(m.loss_scale.cpu())))
                else:
                    # m.loss_scale = loss_scale_factor
                    # if epoch == 10:
                    m.loss_scale.fill_(loss_scale_factor)
                    print("1 module name: " + str(name) + "; loss scale: " + str(np.log2(m.loss_scale.cpu())))
            if self.update_exp_bit_width:
                if self.collect_stats_online:
                    pass
                else:
                    total_bits_num = 8
                    update_from_epoch = self.first_update_with_stats_from_epoch if epoch < self.first_update_with_stats_from_epoch else epoch
                    # pdb.set_trace()
                    exp_std = self.stats_collection.loc[update_from_epoch]['grad std ' + name]
                    # if exp_std > 2.2:
                    #     exp_bits = total_bits_num - 2
                    # else:
                    #     exp_bits = total_bits_num - 3
                    # exp_bits = total_bits_num - 3
                    exp_bits = 3
                    m.fp8_grad_args = dict(exp_width=exp_bits, man_width=int((total_bits_num-1-exp_bits)), exp_bias=(2**(exp_bits-1))-1, roundingMode=0, lfsrVal=0)
                    print("1 module name: " + str(name) + "; fp grad quant: " + str(m.fp8_grad_args) + "; exp_std: " + str(exp_std))
                    # pdb.set_trace()

                # TODO: code it
        # pdb.set_trace()

    def update_stats(self, new_stats):
        if not self.is_enable():
            return
            # def schedule_after_epoch(self, new_stats=None):
    #     if self.scheduled_instructions['collect_stat']:
        self.stats_collection = new_stats.results