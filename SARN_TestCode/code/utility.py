import os
import math
import time
import datetime
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc as misc

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()#记录开始的时间

    def toc(self):
        return time.time() - self.t0 #记录整个过程的时间

    def hold(self):
        self.acc += self.toc() #将这段时间加到self.acc上

    def release(self):
        ret = self.acc
        self.acc = 0 #将总时间赋给ret，清空self.acc计时器

        return ret #返回总时间

    def reset(self):
        self.acc = 0

class checkpoint():
    # args.load是'.'时checkpoint_dir是../SR/BI/RCAN
    # args.load不是'.'时checkpoint_dir是../experiment/args.load，并输出Continue from epoch {？}...
    # 若args.reset为true，则删除checkpoint_dir下所有文件，重新开始训练。
    # 没有checkpoint_dir就新建，再新建checkpoint_dir/Set5/x4
    # 将args中的信息以arg：属性值的形式写入checkpoint_dir配置文件中
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../SR/' + args.degradation + '/' + args.save
            # ../SR/BI/RCAN
        else:
            self.dir = '../experiment/' + args.load # ../experiment/要加载的模型
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir) #删除当前路径下所有文件
            args.load = '.'

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        
        _make_dir(self.dir + '/' + args.testset + '/x' + str(args.scale[0]))
        # self.dir/Set5/x4

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)#若有日志则打开，没有就写入
        # self.log_file=../SR/BI/RCAN/log.txt
        with open(self.dir + '/config.txt', open_type) as f:#打开配置文件
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
            # 将args中的信息以arg：属性值的形式写入self.dir中

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.dir, epoch, is_best=is_best) #保存模型
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )

    def add_log(self, log):
        self.log = torch.cat([self.log, log]) # 将log添加到../SR/BI/RCAN/psnr_log.pt

    # print log，并将log写入模型日志文件，如要刷新，则关闭日志文件后重新打开一遍
    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')#将该log\n写入../SR/BI/RCAN/log.txt
        if refresh:#关闭../SR/BI/RCAN/log.txt后重新打开log文件
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate(self.args.scale):
            plt.plot(
                axis,
                self.log[:, idx_scale].numpy(),
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale):
        filename = '{}/results/{}_x{}_'.format(self.dir, filename, scale)
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range)
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            misc.imsave('{}{}.png'.format(filename, p), ndarr)

    #将文件名中的LRBI或LRBD替换为test，将save_list中的sr、hr、lr图像依次保存到../SR/BI/RCAN/Set5/x4/filename testSR/LR/HR.png
    def save_results_nopostfix(self, filename, save_list, scale):
        #print(filename)
        if self.args.degradation == 'BI':
            filename = filename.replace("LRBI", self.args.save) #将LRBI替换为test
        elif self.args.degradation == 'BD':
            filename = filename.replace("LRBD", self.args.save) #将LRBD替换为test
        
        filename = '{}/{}/x{}/{}'.format(self.dir, self.args.testset, scale, filename)
        # ../SR/BI/RCAN/Set5/x4/filename
        postfix = ('SR', 'LR', 'HR')
        for v, p in zip(save_list, postfix):
            normalized = v[0].data.mul(255 / self.args.rgb_range) #将SR图像重新规整到255范围上
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            #先变成byte类型张量，[height,width,channel],放在CPU上，转换为np类型
            misc.imsave('{}.png'.format(filename), ndarr) #将array保存为图像


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    '''
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else: 
        shave = scale + 6
    '''
    shave = scale
    if diff.size(1) > 1: #若为RGB图像，size=3
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256) #Y = 16 + 1/256 [65.738, 129.057, 25.0647](R, G, B) 即将diff转换为Y通道
        diff = diff.sum(dim=1, keepdim=True) #返回一个张量[1,1,1,1],其中第二维是求和后的结果

    valid = diff[:, :, shave:-shave, shave:-shave] #截掉shave大小的一圈像素
    mse = valid.pow(2).mean() #平方后均值得到MSE值

    return -10 * math.log10(mse)  #得到PSNR值

#返回指定的优化器
def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    #将要求梯度的参数保留下来，不要求梯度的剔除掉
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay
    
    return optimizer_function(trainable, **kwargs)

#做一个学习率衰减函数，使学习率按照指定格式衰减
def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':#学习率衰减函数，每args.lr_decay个epoch，lr乘一次args.gamma
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0) #移除第一个元素，并返回被移除的元素的值
        milestones = list(map(lambda x: int(x), milestones))
        #将milestones中每一个元素都变为int型并排为列表格式，例如[30,80]
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        ) #指epoch>30时，lr*args.gamma，epoch>80,lr*args.gamma,分段常数

    return scheduler

