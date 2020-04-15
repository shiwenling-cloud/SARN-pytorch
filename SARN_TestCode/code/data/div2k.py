import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)
        # 1000 // (800//16)=20 指重复20个epoch测试一次
    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        # [ [] ]，因为self.scale只有一个数，所以循环一次，self.scale应该可以设置为一个列表
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train #训练时一个epoch的图片数
        else:
            idx_begin = self.args.n_train #训练集数800
            idx_end = self.args.offset_val + self.args.n_val # 验证index偏移800+验证集数10
        #这段就是把数据集的图片依次添加到list_hr和list_lr中，其中可以设置self.scale为一个列表，
        #有多个scale比如[2,3,4],然后list_lr变成四维张量
        # [[scale为2的LR图片依次排列][scale为3...][scale为4...]]
        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i) # 比如i=1时，filename=0001
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            # 数据集目录/DIV2K_train_HR/0001.png,...将训练集图片路径依次加到list_hr列表中
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
            # 将数据集目录/DIV2K_train_LR_bicubic/X4/0001x4.png依次添加到列表list_lr[0]中
        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/DIV2K' #数据集目录/DIV2K
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        # 数据集目录/DIV2K_train_HR
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        # 数据集目录/DIV2K_train_LR_bicubic
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )
        #数据集目录/DIV2K/bin/train_bin_HR.npy

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )
        #数据集目录/DIV2K/bin/train_bin_LR_X4.npy
    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

