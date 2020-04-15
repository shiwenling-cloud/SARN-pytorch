import torch
import os

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)

#新建checkpoint_dir和checkpoint_dir/Set5/x4，
# 将args中的信息以arg：属性值的形式写入checkpoint_dir配置文件中
checkpoint = utility.checkpoint(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if checkpoint.ok:
    loader = data.Data(args) #准备数据
    model = model.Model(args, checkpoint) #恢复出要用的模型
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    #返回loss_sum,如果是测试的话就不计算loss反向传播更新参数了，应该是另外计算loss的
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

