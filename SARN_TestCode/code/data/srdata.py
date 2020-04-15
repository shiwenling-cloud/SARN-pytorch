import os

from data import common

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data) #数据集目录

        def _load_bin():
            self.images_hr = np.load(self._name_hrbin())
            #从数据集目录/DIV2K/bin/train_bin_HR.npy中加载出HR图像
            self.images_lr = [
                np.load(self._name_lrbin(s)) for s in self.scale
            ] #从数据集目录/DIV2K/bin/train_bin_LR_X4.npy中加载出LR图像
            #若self.scale为列表[2,3,4],则分别加载出X2,X3,X4的LR图像到self.images_lr

        if args.ext == 'img' or benchmark:
            self.images_hr, self.images_lr = self._scan()
        elif args.ext.find('sep') >= 0:
            self.images_hr, self.images_lr = self._scan()
            #若args.ext设置为sep_reset，则将self.images_hr和 self.images_lr中的图片转换为.npy格式
            if args.ext.find('reset') >= 0:
                print('Preparing seperated binary files')
                for v in self.images_hr:
                    hr = misc.imread(v) #将图片读取出来为array类型，即numpy类型
                    name_sep = v.replace(self.ext, '.npy')#.npy文件是numpy专用的二进制文件
                    np.save(name_sep, hr) #将numpy类型的图片保存为npy文件
                for si, s in enumerate(self.scale):
                    for v in self.images_lr[si]:
                        lr = misc.imread(v)
                        name_sep = v.replace(self.ext, '.npy')
                        np.save(name_sep, lr)

            self.images_hr = [
                v.replace(self.ext, '.npy') for v in self.images_hr
            ]
            self.images_lr = [
                [v.replace(self.ext, '.npy') for v in self.images_lr[i]]
                for i in range(len(self.scale))
            ]

        elif args.ext.find('bin') >= 0: #图片已经是二进制文件
            try:
                if args.ext.find('reset') >= 0:#已经导入的是二进制文件，不需要reset设置
                    raise IOError
                print('Loading a binary file')
                _load_bin()
                #分别从self._name_hrbin、self._name_lrbin中加载出npy文件存到self.images_hr、self.images_lr
            except:
                #若文件不是二进制文件npy，则从self._scan()中导入图片到list_hr, list_lr，然后将
                #列表中的文件转换为npy文件存到self._name_hrbin()和 self._name_lrbin中
                print('Preparing a binary file')
                bin_path = os.path.join(self.apath, 'bin')#数据集目录/DIV2K/bin
                if not os.path.isdir(bin_path):
                    os.mkdir(bin_path)

                list_hr, list_lr = self._scan()
                hr = [misc.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr) #数据集目录/DIV2K/bin/train_bin_HR.npy
                del hr
                for si, s in enumerate(self.scale):
                    lr_scale = [misc.imread(f) for f in list_lr[si]]
                    np.save(self._name_lrbin(s), lr_scale)
                    del lr_scale
                _load_bin()
        else:
            print('Please define data type')

    def _scan(self):
        raise NotImplementedError
    # 在面向对象编程中，可以先预留一个方法接口不实现，在其子类中实现。如果要求其子类一定要实现，
    # 不实现的时候会导致问题，那么采用raise的方式就很好。而此时产生的问题分类是NotImplementedError。

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def _name_lrbin(self, scale):
        raise NotImplementedError

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr, hr = common.set_channel([lr, hr], self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor([lr, hr], self.args.rgb_range)
        return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = hr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            lr = np.load(lr)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, filename

    def _get_patch(self, lr, hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = common.augment([lr, hr])
            lr = common.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

