import os
from importlib import import_module

import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__() # nn.Module
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        module = import_module('model.' + args.model.lower()) # 导入模块model.edsr
        self.model = module.make_model(args).to(self.device) #构建好模型并分配到CPU设备
        #torch.device代表将torch.Tensor分配到的设备的对象，这里是CPU
        if args.precision == 'half': self.model.half() #将tensor投射为半精度浮点类型

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs)) #多GPU训练

        #恢复出要用的模型，要么预训练的模型（args.pre_train），
        # 要么RCAN中上次或第几次保存的模型（args.resume）
        self.load(
            ckp.dir, #../SR/BI/RCAN
            pre_train=args.pre_train, #预训练模型目录
            resume=args.resume, #从特定checkpoint中恢复
            cpu=args.cpu # 只用CPU
        )
        if args.print_model: print(self.model)

    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale #scale下标号
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)#设置scale

        if self.self_ensemble and not self.training:
            if self.chop:
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    #将模型的state_dict保存到../SR/BI/RCAN/model/model_latest.pt、../SR/BI/RCAN/model/model_best.pt、../SR/BI/RCAN/model/model_epoch数.pt
    def save(self, apath, epoch, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model', 'model_latest.pt')
        ) #将self.model.state_dict()存到../SR/BI/RCAN/model/model_latest.pt中去
        if is_best:#看best[1][0] + 1 == epoch
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )#若是best结果，则存到../SR/BI/RCAN/model/model_best.pt
        
        if self.save_models:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            ) #将self.model.state_dict()存到../SR/BI/RCAN/model/model_epoch数.pt中

    def load(self, apath, pre_train='.', resume=-1, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
            # pytorch将gpu训练好的模型参数load到cpu上，gpu -> cpu
        else:
            kwargs = {}

        if resume == -1: #不从checkpoint中恢复
            # 从../SR/BI/RCAN/model/model_latest.pt中恢复出训练好的模型参数，得到其state_dict()
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_latest.pt'),
                    **kwargs
                ),
                strict=False
            )
        # self.get_model()是恢复的模型框架，
        # .load_state_dict(torch.load(pre_train, **kwargs),strict=False)从特定checkpoint文件中
        # 恢复出预训练的模型state_dict即每层与对应参数的映射关系，这样就将预训练好的参数应用在了
        #恢复的模型中，得到了预训练的模型整体。
        elif resume == 0:
            if pre_train != '.':
                print('Loading model from {}'.format(pre_train))
                self.get_model().load_state_dict(
                    torch.load(pre_train, **kwargs),
                    strict=False
                ) #恢复预训练模型的state_dict到CPU中
        else:
            # 恢复../SR/BI/RCAN/model/model_{resume}.pt中的模型
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model', 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=False
            )
    # shave = 10, min_size=160000
    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale] #要用的scale
        n_GPUs = min(self.n_GPUs, 4) #最多四个GPU
        b, c, h, w = x.size() #不懂这里为啥是[batchsize,channel,height,width]了
        h_half, w_half = h // 2, w // 2 # h=w=48
        h_size, w_size = h_half + shave, w_half + shave # h_size=w_size=34
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]#进行数据chop，这样一张LR图像就变成4张了，数据扩增

        if w_size * h_size < min_size:#像素量不能超过min_size
            sr_list = []
            for i in range(0, 4, n_GPUs):#比如n_GPUs=2时，lr_batch就是lr_list前两个list在0维度concat后
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0) #
                sr_batch = self.model(lr_batch) #通过模型得到预测输出sr_batch
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))#按照n_GPUs数分块写入sr_list,即sr_list跟lr_list的格式是对应的
        else:#若每幅lr图的像素量太大超过min_size，则将其进一步chop，一张变成四张，这样就不大了
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w #48*4
        h_half, w_half = scale * h_half, scale * w_half #24*4
        h_size, w_size = scale * h_size, scale * w_size # 34*4
        shave *= scale #10*4

        output = x.new(b, c, h, w) #新建一个output tensor，用来存放sr
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output
#之前处理lr_list的时候加了shave，每个lr是34，之后通过模型放大四倍时变成34*4大小，最后拼接成最终SR图像时会根据位置减去多余的部分，
#变成24*4，然而再根据lr chop时的位置拼接到对应的位置。

    def forward_x8(self, x, forward_function):
        def _transform(v, op):#返回水平、垂直、对角线翻转后的图像
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()#将v variable中的tensor取出来放在CPU上，并将tensor转换为numpy格式
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy() #[batchsize,channel,height,width]，即将width倒序输出，水平180度翻转
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy() #垂直翻转
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy() #沿对角线翻转

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list]) #四个batch的lr图像

        sr_list = [forward_function(aug) for aug in lr_list] #得到四个batch的sr图像
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't') #旋转
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h') #垂直翻转+垂直翻转=原图
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v') #水平翻转+水平翻转=原图  对角线翻转+水平翻转=旋转90度

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output
        #返回逆翻转后且求平均后的输出

