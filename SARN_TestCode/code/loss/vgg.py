from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features #调用VGG19预训练模型
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8]) #VGG19前8层的特征
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        #定义了一个数据归一化操作，(img - vgg_mean) / vgg_std
        self.vgg.requires_grad = False

    #求vgg的感知损失
    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)#归一化
            x = self.vgg(x)#通过vgg
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
            #返回一个不参与计算图的Tensor y

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
