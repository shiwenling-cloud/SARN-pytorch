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
    #返回排序好的测试文件路径列表list_hr, list_lr
    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):#遍历目录
            filename = os.path.splitext(entry.name)[0] #分离文件名和扩展名，并取第一个分片即文件名
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            #将路径数据集目录/benchmark/测试数据集名/HR/文件名.png依次存到list_hr中
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        list_hr.sort() #对数组进行排序
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        # 数据集目录/benchmark/测试数据集名
        self.dir_hr = os.path.join(self.apath, 'HR')#数据集目录/benchmark/测试数据集名/HR
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')#数据集目录/benchmark/测试数据集名/LR_bicubic
        self.ext = '.png'
