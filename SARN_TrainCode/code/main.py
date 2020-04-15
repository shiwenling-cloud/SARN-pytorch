import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.nn as nn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#rcan_rdn_3_6
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    '''
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')  # 卷积层参数初始化
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_()  # 全连接层参数初始化
    '''
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate(): #判断要不要结束
        t.train()#训练
        t.test() #验证

    checkpoint.done()

