from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate

#将训练图片、测试图片加载出来整理为batch的格式，
class Data:
    def __init__(self, args):
        kwargs = {} # kwargs就是当你传入key=value是存储的字典，成对键值对
        if not args.cpu:
            kwargs['collate_fn'] = default_collate #输入batch_size张图片，组成一个batch张量
            kwargs['pin_memory'] = True #若不是只用CPU（还用GPU），则设置锁页内存
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False #若只用CPU，则设置不锁页内存

        self.loader_train = None
        if not args.test_only: #如果不是测试，即为训练
            module_train = import_module('data.' + args.data_train.lower())
            # 导入模块data.DIV2K
            trainset = getattr(module_train, args.data_train)(args)#等效于调用module_train.DIV2K
            #即trainset=div2k中DIV2K类的一个例子
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
            #将trainset中数据分块为batch
        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
                # testset=Benchmark(args, train=False)返回测试文件路径列表list_hr, list_lr
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )

        else:#如果是DIV2K，就不能用benchmark了，用它特定的函数
            module_test = import_module('data.' +  args.data_test.lower())#data.div2k
            testset = getattr(module_test, args.data_test)(args, train=False)
            #testset=DIV2K(args, train=False)
        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
        # 将testset(此时列表中是文件路径)中数据加载出来分块为batch输出