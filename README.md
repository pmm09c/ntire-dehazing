# Feature Forwarding for Efficient Image Dehazing
Method for 2019 CVPR NITRE Workshop, Image Dehazing Challenge

## NITRE 2019 Image Haze Removal Challenge
[Competition Page](https://competitions.codalab.org/competitions/21163)

### Entry Information

#### Primary References

[e-lab. LinkNet](https://github.com/e-lab/pytorch-linknet)

[He, Zhang. Densely Connected Pyramid Dehazing Network](https://github.com/hezhangsprinter/DCPDN)

#### Install Our Models

The code has been developed and tested with pytorch 1.0+ in a conda environment

```bash
conda activate {Project Environment}
git clone ${repository}
cd nitre-dehazing
conda install --yes --file requirements.txt
cd ..
git clone https://github.com/Po-Hsun-Su/pytorch-ssim
git clone https://github.com/jorge-pessoa/pytorch-msssim
cd pytorch-ssim 
python setup.py
cd ../pytorch-msssim
python setup.py
```

#### Data

Our competition results can be recreated by training with the 2019 NITRE Image Dehazing Dataset.

Other datasets available for image dehazing: [I-HAZE](http://www.vision.ee.ethz.ch/ntire18/i-haze/), [O-HAZE](http://www.vision.ee.ethz.ch/ntire18/o-haze/), and [He, Zhang's training dataset](https://github.com/hezhangsprinter/DCPDN).

#### Training Our Models
We propose two approaches. One approach estimates airlight and transmission maps with two separate encoder-to-decoder networks that feed into a refinement layer. This model is named DualFastNet. The second approach uses a single network to a refinement layer. This model is named FastNet (uses ResNet 18 pretrained model) / FastNet50 (uses ResNet 50 pretrained model). These models draw upon past work, including [DCPDN](https://github.com/hezhangsprinter/DCPDN) and [LinkNet](https://github.com/e-lab/pytorch-linknet).

To train:
```bash
# From randomly initialized weights
python train.py config.json
# From randomly pretrained weights
python train.py config.json {path_to_weights_file}
```

All train configs can be found in configs/train and modified as needed. Config json files are available for training a separate airlight model, a separate transmission map model, a DualFastNet model,a FastNet model, or a FastNet50 model. Each config is available for both the 2019 NITRE Image Dehazing dataset or the He, Zhang dataset.

##### Loss Functions
Our JSON configuration files natively support the following loss functions: L1 Loss, MSE Loss, BCE Loss, Huber Loss, SSIM Loss, MSSSIM Loss, and Content Loss. Content Loss is computed as described in[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). Multiple losses can be used with weightings specified in the JSON configuration files. For example to use 10xMSE and 1xSSIM, you can specify the following in the config:
```bash
"loss_image": ["MSE","SSIM"],
"loss_image_w": [10.0,1.0],
```

#### Testing Our Models 
As in training, testing can be done using a JSON configuration file. These are located in configs/test and can be modified as needed.
```bash
# To Test
python test.py config.json {path_to_weights_file}
``` 
#### Results
![Test Image](https://github.com/pmm09c/nitre-dehazing/blob/master/dataset/test/53.png "Example Test Image")
![Test Dehazed Image](https://github.com/pmm09c/nitre-dehazing/blob/master/results_nitre/2.png "Example Test Image")
