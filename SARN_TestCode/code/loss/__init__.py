import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#将args.loss中要求的各loss分别构建出模块加到self.loss列表中，loss_function加到self.loss_module
# 将各self.loss中的loss分别作用与sr和hr得到损失值列表losses，加和得到loss_sum返回。
class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*') # weight=1，loss_type=L1
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()#MSE损失
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()#L1损失
            elif loss_type.find('VGG') >= 0:#loss_type应该是VGG22或VGG54，所以loss_type[3:]就是指22或者54
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )#返回vgg特征损失
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                ) #返回生成器损失
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})
            #如果是GAN的话，有两个loss都要加进去，DIS就是判别器的损失
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))#weight保留小数点后三位
                self.loss_module.append(l['function'])#将loss加入self.loss_module模块

        self.log = torch.Tensor()#没有提供参数，返回一个零维张量

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        #torch.device代表将self.loss_module分配到的设备的对象，这里是CPU
        if args.precision == 'half': self.loss_module.half()#将tensor投射为半精度浮点类型
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            ) #多GPU训练

        if args.load != '.': self.load(ckp.dir, cpu=args.cpu)
        #checkpoint_dir是.. / experiment / args.load，并输出Continue from epoch {？}...

    def forward(self, sr, hr):
        losses = []
        #将各个loss依项作用sr和hr后加起来
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()#取元素值加到self.log[-1, i]中
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss
                #adversarial中的self.loss即判别器损失

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item() #在行末位-列末位的位置加上loss_sum的元素值

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    #在原来log tensor下加一行零，不知道要干嘛
    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))
        #例若损失函数为GAN和L2，则初始化零维张量self.log为[0,0,0,0]

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)
        #计算总loss对总batch的平均值

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
        #列表元素依次相连，即将self.loss和self.log中对应的loss值相连起来，self.log=[gen,dis,L2,sum]（若损失函数为GAN和L2）
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples)) #输出[loss类型： 该loss对batch数的平均值]

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('{}/loss_{}.pdf'.format(apath, l['type']))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))
    #将要加载模型的loss模块参数加载出来并根据epoch调整学习率
    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
            #将张量加载到cpu上
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),#加载../experiment/args.load/loss.pt参数到cpu上
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        #加载../experiment/args.load/loss_log.pt到self.log
        for l in self.loss_module:#各loss function
            if hasattr(l, 'scheduler'): #就是self.scheduler
                for _ in range(len(self.log)): l.scheduler.step()
                #将scheduler.step()用在epoch内可以调整lr

