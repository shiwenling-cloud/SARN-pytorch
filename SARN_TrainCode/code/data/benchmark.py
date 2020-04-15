import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry_hr in os.scandir(self.dir_hr):
            filename_hr = os.path.splitext(entry_hr.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename_hr + self.ext))
        for entry_lr in os.scandir(self.dir_lr):
            filename_lr = os.path.splitext(entry_lr.name)[0]
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}{}'.format(filename_lr, self.ext)
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR','x%d'%self.scale[0])
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic','x%d'%self.scale[0])
        self.ext = '.png'
