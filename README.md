# Feature Forwarding for Efficient Image Dehazing
Method for 2019 CVPR NITRE Workshop, Image Dehazing Challenge

## NITRE 2019 Image Haze Removal Challenge
[Competition Page](https://competitions.codalab.org/competitions/21163)

### Entry Information

#### Primary References

[e-lab. LinkNet](https://github.com/e-lab/pytorch-linknet)

[He, Zhang. Densely Connected Pyramid Dehazing Network](https://github.com/hezhangsprinter/DCPDN)

#### Install and Data

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

Datasets used were provided by NITRE and the He, Zhang training Dataset found at https://github.com/hezhangsprinter/DCPDN.

#### Training and Test
We propose two network architectures: FastNet and DualFastNet. FastNet utilizes a single LinkNet for its encoder-decoder, while DualFastNet utilizes two LinkNet encoder-decoders to learn transmission map and atmospheric light estimations. Both networks include refinement layers inspired by Zhang He's DCPDN network.

##### Training Atmospheric Light Model
```bash
# Train
python train.py configs/train_atmos.py
# Test
python test.py configs/test_atmos.py {airlight weights}
```
##### Training Transmission Map Model
```bash
# Train
python train.py configs/train_trans.py
# Test
python test.py configs/test_trans.py {trans weights}
``` 
##### Training DualFastNet Model on He Zhang Dataset
```bash
# Train
python train.py configs/train_full.py {trans weights} {airlight weights} 
# Test
python test.py configs/test_full.py {full weights}
``` 
##### Fine Tuning DualFastNet Model on NITRE Dataset
```bash
# Train
python train.py configs/train_nitre.py {full weights} 
# Test
python test.py configs/test_nitre.py {nitre weights}
``` 
#### Results
![Test Image](https://github.com/pmm09c/nitre-dehazing/blob/master/dataset/test/53.png "Example Test Image")
![Test Dehazed Image](https://github.com/pmm09c/nitre-dehazing/blob/master/results_nitre/2.png "Example Test Image")
