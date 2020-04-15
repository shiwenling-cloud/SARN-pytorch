import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import model
import utility
from option import args
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = {}
        for name, module in self.submodule._modules.items():
            if name in ['body']:
                x0 = x
                for name, module in module._modules.items():
                    if name in ['0']:
                        x1 = x
                        for name, module in module._modules.items():
                            for name, module in module._modules.items():
                                if name in ['5']:
                                    x2 = x
                                    for name, module in module._modules.items():
                                        '''
                                        if name in ['ck']:
                                            x3 = x
                                            outputs['conv2'] = x3
                                            for name, module in module._modules.items():
                                                for name, module in module._modules.items():
                                                    x = module(x)
                                                    print(name)
                                                    if self.extracted_layers is None or name in self.extracted_layers:
                                                          outputs[name] = x * x3

                                            x = x3 * x
                                            #outputs['ck'] = x
                                    
                                        else:
                                            x = module(x)
                                            #print(name)
                                        '''
                                        x = module(x)
                                        print(name)
                                        if self.extracted_layers is None or name in self.extracted_layers:
                                            outputs[name] = x + x2

                                    x = x + x2
                                else:
                                    x = module(x)
                                    # print(name)
                        x = x + x1
                    else:
                        x = module(x)
                        # print(name)
                x = x + x0
            else:
                x = module(x)
                # print(name)
        #outputs['out'] = x


        return outputs


def get_picture(pic_name, transform):
    img = skimage.io.imread(pic_name)
    img = skimage.transform.resize(img, (128, 128))
    img = np.asarray(img, dtype=np.float32)
    return transform(img)


def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)


def get_feature():
    pic_dir = './img_012_SRF_4_LR.png'
    transform = transforms.ToTensor()
    img = get_picture(pic_dir, transform)
    print(img.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 插入维度
    img = img.unsqueeze(0)
    img = img.to(device)
    # img = F.interpolate(img, scale_factor=0.25, mode='bicubic')
    net = model.Model(args, checkpoint).to(device)
    # net.load_state_dict(torch.load('E:\研一\研一下\tensorflow\RCAN\RCAN-master - rdn\RCAN_TestCode\model\sarn_x4.pt'))
    exact_list = ['conv2','ck']  # 要提取层的name列表
    dst = './features-sarn_3_6/RG1_RCAB6'
    therd_size = 256

    myexactor = FeatureExtractor(net.model, exact_list)
    outs = myexactor(img)  # 一个放着各层name：feature的列表
    for k, v in outs.items():
        features = v[0]  # [batchsize, C, H, W]-->[C, H, W]
        # print(features.shape)
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        #iter_range = features.shape[0] #channel数
        # print(feature_mean.shape)
        #for i in range(iter_range):
        # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')

        feature = feature_mean.data.cpu().numpy()
            # print(feature.shape)
        feature_img = feature[0, :, :]
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

        dst_path = os.path.join(dst, k)

        make_dirs(dst_path)
            #feature_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_HSV)
            # 若要存的图比[256, 256]小的话，就resize到[256, 256]大小，存到./features/name/通道index_256.png
        if feature_img.shape[0] < therd_size:
            tmp_file = os.path.join(dst_path, str(0) + '_' + str(therd_size) + '.png')
            tmp_img = feature_img.copy()
            tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(tmp_file, tmp_img)
                            # 若要存的图比256大，则直接存到./features/name/通道index.png
        dst_file = os.path.join(dst_path, str(0) + '.png')
        cv2.imwrite(dst_file, feature_img)


if __name__ == '__main__':
    get_feature()