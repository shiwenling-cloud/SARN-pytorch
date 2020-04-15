import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train #训练数据集分块处理好的batch图像
        self.loader_test = loader.loader_test #测试集图像
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )#加载checkpoint的优化器state_dict
            for _ in range(len(ckp.log)): self.scheduler.step()#在epoch内更新lr

        self.error_last = 1e8

    def train(self):
        self.scheduler.step() #调整学习率
        self.loss.step() #构建loss模块
        epoch = self.scheduler.last_epoch + 1 #int整型，现在的epoch数
        lr = self.scheduler.get_lr()[0] #输出最后一次调整后学习率的值，0应该指第一个值，可能是先进后出的队列形式

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )#print [Epoch epoch数] Learning rate:将lr转换为小数并用指数记法表示出来
        #并将log存到../SR/BI/RCAN/log.txt中
        self.loss.start_log()#例若损失函数为GAN和L2，则[[gen,dis,L2,sum][0,0,0,0]]
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer() #开始时间
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            #依次处理训练数据集分块处理好的batch图像，batch为batch的计数下标，
            lr, hr = self.prepare([lr, hr]) #变成半精度浮点数以及发送到指定设备（cpu或者cuda）
            timer_data.hold()#记录间隔时间
            timer_model.tic()#开始计时

            self.optimizer.zero_grad()#直接把模型的参数梯度设成0：
            sr = self.model(lr, idx_scale) #得到预测输出
            loss = self.loss(sr, hr) #返回sr和hr的总loss
            if loss.item() < self.args.skip_threshold * self.error_last:
                #要保证sr和hr的loss不能大于skip_threshold,否则就跳过，即跳过loss大于某门槛的batch
                loss.backward() #反向传播求梯度
                self.optimizer.step() #优化器更新梯度，即更新模型
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))#输出跳过的batch序号，以及其loss

            timer_model.hold() #得到处理一个batch的间隔时间

            if (batch + 1) % self.args.print_every == 0:#若batch数已经累积到可以log训练状态时
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch), #依次输出[loss类型: loss对batch平均值]
                    timer_model.release(),
                    timer_data.release()))
                #[图片数/数据集总图片数 依次输出[loss类型: loss对batch平均值]] 模型处理一个batch数据时间+获取该batch数据时间s

            timer_data.tic() #开始计时获取数据时间

        self.loss.end_log(len(self.loader_train)) #self.log最后一位loss.sum计算对总batch数的平均值
        self.error_last = self.loss.log[-1, -1] #总loss平均值赋给self.error_last

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        #在../SR/BI/RCAN/psnr_log.pt tensor中添加self.scale长度的一行0，作为一会儿记录psnr值的初始化过程
        self.model.eval() #运行模型

        timer_test = utility.timer() #开始计时
        with torch.no_grad():#表示下面过程不进行反向传播，即不求梯度
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale) #设置要测试的scale
                tqdm_test = tqdm(self.loader_test, ncols=80) #加载测试数据迭代进度条
                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):#依次处理测试图像
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    #若hr为单通道图像，则no_eval=0,则直接处理
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:#若hr为多通道图像，则只用lr的第一个channel的图像，即Y通道
                        lr = self.prepare([lr])[0]

                    sr = self.model(lr, idx_scale) #得到sr输出
                    sr = utility.quantize(sr, self.args.rgb_range) #数字转换到args.rgb_range

                    save_list = [sr] #应该也是[batchsize,channel,height,width]
                    if not no_eval:#hr为单通道
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        ) #计算准确度
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        #self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_nopostfix(filename, save_list, scale)
                        #将得到的结果保存到../SR/BI/RCAN/Set5/x4目录下

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                #往../SR/BI/RCAN/psnr_log.pt中记录一条PSNR log = PSNR平均值
                best = self.ckp.log.max(0) #选取self.ckp.log每一列最大值组成的一维数组，每个scale对应的最大PSNR值
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale], #测试平均PSNR值
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )#该scale下输出测试集结果

        self.ckp.write_log(
            'Total time: {:.2f}s, ave time: {:.2f}s\n'.format(timer_test.toc(), timer_test.toc()/len(self.loader_test)), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))
    #将l中的tensor变成半精度浮点数（若self.args.precision == 'half'）以及用指定的设备（cpu或者cuda），返回处理后的列表
    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

