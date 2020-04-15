import utility
from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP':#判断是不是WGAN_GP，不是就按照指定方式优化，是的话就用Adam优化。
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utility.make_scheduler(args, self.optimizer)#学习率衰减器

    def forward(self, fake, real):
        fake_detach = fake.detach()#梯度截断，该参数往下的支流不求梯度

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()#把判别器的梯度初始化为零
            d_fake = self.discriminator(fake_detach)
            #G生成的fake图让D来判别，得到的损失，计算梯度进行反传。这个梯度只能影响G，不能影响D！
            #所以这里要设置fake.detach()，使下面判别器反向传播更新参数时不求fake_detach支流的梯度。
            d_real = self.discriminator(real)#真实图像判别器要求导优化，使判别真实图像的几率最大
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)#返回一个用标量值 0 填充的张量,其大小与输入相同
                label_real = torch.ones_like(d_real)#返回一个用标量值 1 填充的张量,其大小与输入相同
                loss_d \
                    = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                    + F.binary_cross_entropy_with_logits(d_real, label_real)
                #损失函数应该让d_fake与label_fake分布、d_real与label_real分布尽可能相近
                # D(x)-->1,D(G(x))-->0
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()#真和假之间的差距均值 D(G(x))-D(x)均值尽可能小
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    #返回一个与fake相同size的向量，用[0,1]间均匀分布随机数填充，[16*48*4*48*4*64,1,1,1]
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    # [16,48*4,48*4,64]
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()#得到loss_d元素值，计算判别器loss
            loss_d.backward()#反向传播
            self.optimizer.step()#更新模型

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)#将p中的元素限制在[-1,1]范围内并返回一个Tensor

        self.loss /= self.gan_k
        #生成器loss
        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            ) #D(G(x))-->1
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()
            #D(G(x))均值尽可能大
        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
