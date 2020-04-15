# Image Super-Resolution Using Very Deep Residual Channel Attention Networks
This repository is for SARN introduced in the following paper

[shiwenling_cloud](http://shiwenling_cloud.com/), "(SARN)Spatial-wise Attention Residual Network for Image Super-resolution"

The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 14.04/16.04 environment (Python3.6, PyTorch_0.4.0, CUDA8.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs. RCAN model has also been merged into [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Recent research suggests that attention mechanism is capable of improving performance of deep learning based single image super-resolution (SISR) methods. In this work, we propose a deep spatial-wise attention residual network (SARN) for SISR. Specifically, we propose a novel spatial attention block (SAB) to rescale pixel-wise features by explicitly modeling interdependencies between pixels on each feature map, encoding where (i.e., attentive spatial pixels in feature map) the visual attention is. A modified patch-based non local block (p-NLB) can be inserted in SAB to capture long-distance spatial contextual information and relax the local neighborhood constraint. Furthermore, we design a bottleneck spatial attention module (BSAM) to widen the network so that more information is allowed to pass. Meanwhile, we adopt local and global residual connections in SISR to make the network focus on learning valuable high-frequency information. Extensive experiments show the superiority of the proposed SARN over the state-of-art methods on benchmark datasets both accuracy and visual quality

![SAB](/Figs/SAB.pdf)
Spatial attention block(SAB).
![BSAM](/Figs/BSAM.pdf)
Bottleneck spatial attention module (BSAM).
![SARN](/Figs/SARN.pdf)
The architecture of our proposed spatial attention residual network (SARN).

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download models for our paper and place them in '/RCAN_TrainCode/experiment/model'.

    All the models (BIX2/3/4) can be downloaded [BaiduYun](https://pan.baidu.com/s/1_Qx4NGYH1hJ0um9zKNbrwQ). Password:2nbg

2. Cd to 'SARN_TrainCode/code', run the following scripts to train models.

    **You can use scripts in file 'TrainRCAN_scripts' to train models for our paper.**

    ```bash
    # BI, scale 2, 3, 4
    # SARN_BIX2_G7R20P48, input=48x48, output=96x96
    python main.py --model SARN --save SARN_BIX2_G7R20P48 --scale 2 --n_resblocks 7 --n_BSAMs 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96

    # SARN_BIX3_G10R20P48, input=48x48, output=144x144
    python main.py --model SARN --save SARN_BIX3_G10R20P48 --scale 3 --n_resblocks 7 --n_BSAMs 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 144 --pre_train ../experiment/model/sarn_x2.pt

    # SARN_BIX4_G10R20P48, input=48x48, output=192x192
    python main.py --model SARN --save SARN_BIX4_G10R20P48 --scale 4 --n_resblocks 7 --n_BSAMs 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/model/sarn_x2.pt
    ```

## Test
### Quick start
1. Download models for our paper and place them in '/RCAN_TestCode/model'.

    All the models (BIX2/3/4) can be downloaded from [BaiduYun](https://pan.baidu.com/s/1_Qx4NGYH1hJ0um9zKNbrwQ). Password:2nbg

2. Cd to '/SARN_TestCode/code', run the following scripts.

    **You can use scripts in file 'TestSARN_scripts' to produce results for our paper.**

    ```bash
    # No self-ensemble: SARN
    # BI degradation model, X2, X3, X4
    # SARN_BIX2
    python main.py --data_test MyImage --scale 2 --model SARN  --n_resblocks 7 --n_BSAMs 20 --n_feats 64 --pre_train ../model/SARN_BIX2.pt --test_only --save_results --chop --save 'SARN' --testpath ../LR/LRBI --testset Set5
    # SARN_BIX3
    python main.py --data_test MyImage --scale 3 --model SARN  --n_resblocks 7 --n_BSAMs 20 --n_feats 64 --pre_train ../model/SARN_BIX3.pt --test_only --save_results --chop --save 'SARN' --testpath ../LR/LRBI --testset Set5
    # RCAN_BIX4
    python main.py --data_test MyImage --scale 4 --model SARN  --n_resblocks 7 --n_BSAMs 20 --n_feats 64 --pre_train ../model/SARN_BIX4.pt --test_only --save_results --chop --save 'SARN' --testpath ../LR/LRBI --testset Set5
    # With self-ensemble: SARN+
    # SARN+_BIX2
    python main.py --data_test MyImage --scale 2 --model SARN  --n_resblocks 7 --n_BSAMs 20 --n_feats 64 --pre_train ../model/SARN+_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'SARN+' --testpath ../LR/LRBI --testset Set5
    # SARN+_BIX3
    python main.py --data_test MyImage --scale 3 --model SARN  --n_resblocks 7 --n_BSAMs 20 --n_feats 64 --pre_train ../model/SARN+_BIX3.pt --test_only --save_results --chop --self_ensemble --save 'SARN+' --testpath ../LR/LRBI --testset Set5
    # SARN+_BIX4
    python main.py --data_test MyImage --scale 4 --model SARN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/SARN+_BIX4.pt --test_only --save_results --chop --self_ensemble --save 'SARN+' --testpath ../LR/LRBI --testset Set5
    ```

### The whole test pipeline
1. Prepare test data.

    Place the original test sets (e.g., Set5, other test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo?usp=sharing) or [Baidu](https://pan.baidu.com/s/1yBI_-rknXT2lm1UAAB_bag)) in 'OriginalTestData'.

    Run 'Prepare_TestData_HR_LR.m' in Matlab to generate HR/LR images with different degradation models.
2. Conduct image SR. 

    See **Quick start**
3. Evaluate the results.

    Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.

## Results
### Quantitative Results
![PSNR_SSIM_BI](/Figs/psnr_bi.png)
Quantitative results with BI degradation model. Best and second best results are marked in red and green.

### Visual Results
![Visual_PSNR_SSIM_BI](/Figs/Fig1_visual_psnr_ssim_bi_x4.pdf)
![Visual_PSNR_SSIM_BI](/Figs/Fig2_visual_psnr_ssim_bi_x4.pdf)

Visual comparison for 4× SR with BI model


## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [PyTorch version](https://github.com/thstkdgus35/EDSR-PyTorch).

