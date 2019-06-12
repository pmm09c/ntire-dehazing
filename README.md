# Feature Forwarding for Efficient Image Dehazing
Method for 2019 CVPR NTIRE Workshop, Image Dehazing Challenge

Our Paper: [https://arxiv.org/abs/1904.09059](https://arxiv.org/abs/1904.09059)

## NTIRE 2019 Image Haze Removal Challenge
[Competition Page](https://competitions.codalab.org/competitions/21163)

### Entry Information

#### Primary References

[e-lab. LinkNet](https://github.com/e-lab/pytorch-linknet)

[Zhang, He. Densely Connected Pyramid Dehazing Network](https://github.com/hezhangsprinter/DCPDN)

#### Install Our Models

The code has been developed and tested with pytorch 1.0+ in a conda environment

```bash
conda activate {Project Environment}
git clone ${repository}
cd ntire-dehazing
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

Our competition results can be recreated by training with the [Dense-Haze Dataset](https://arxiv.org/abs/1904.02904), introduced in the 2019 NTIRE Image Dehazing Challenge.

Other datasets available for image dehazing: [I-HAZE](http://www.vision.ee.ethz.ch/ntire18/i-haze/), [O-HAZE](http://www.vision.ee.ethz.ch/ntire18/o-haze/), and [He, Zhang's training dataset](https://github.com/hezhangsprinter/DCPDN).

#### Training Our Models
We propose two approaches. One approach estimates airlight and transmission maps with two separate encoder-to-decoder networks that feed into a refinement layer. This model is named DualFastNet. The second approach uses a single network to a refinement layer. This model is named FastNet (uses ResNet 18 pretrained model) / FastNet50 (uses ResNet 50 pretrained model). These models draw upon past work, including [DCPDN](https://github.com/hezhangsprinter/DCPDN) and [LinkNet](https://github.com/e-lab/pytorch-linknet).

##### To Train
```bash
# From randomly initialized weights
python train.py config.json
# From pretrained weights
python train.py config.json {path_to_weights_file}
```

All train configuration JSONs can be found in configs/train and modified as needed. Configuration JSON files are available for training a separate airlight model, a separate transmission map model, a DualFastNet model, a FastNet model, or a FastNet50 model. Each configuration JSON file is available for both the 2019 NTIRE Image Dehazing dataset or the He, Zhang dataset. Validation can also be done during training by setting the "validate" flag to "1" in the JSON configuration file and providing a path to the validation JSON configuration file in the "validation\_config" field, as shown below:

```bash
"validate":1,
"validation_config":"./configs/validation/validate.json",
```

##### Loss Functions
Our JSON configuration files natively support the following loss functions: L1 Loss, MSE Loss, BCE Loss, Huber Loss, SSIM Loss, MSSSIM Loss, PSNR Loss, and Content Loss. Content Loss is computed as described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). Multiple losses can be used with weightings specified in the JSON configuration files. For example to use 10xMSE and 1xSSIM, you can specify the following in the config:
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

#### Recreating 2019 NTIRE Challenge Results
Copy the weights from the release tagged ntire_submission to the wgts directory and run the following command
```bash
python test.py configs/test/test_nitre.json wgts/ntire_submission.ckpt
``` 

#### Results
![Test Image](https://github.com/pmm09c/nitre-dehazing/blob/master/dataset/test/53.png "Example Test Image")
![Test Dehazed Image](https://github.com/pmm09c/nitre-dehazing/blob/master/results_ntire/2.png "Example Test Image")
