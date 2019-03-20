# IRP
Image restoration projects

## NITRE 2019 Image Haze Removal Challenge
[Competition Page](https://competitions.codalab.org/competitions/21163)

### Sample Test Image
![Test Image](https://llcad-github.llan.ll.mit.edu/g46-AI/IRP/blob/master/dataset/test/51.png "Example Test Image")

### Entry Information

#### Primary References

[e-lab. LinkNet](https://github.com/e-lab/pytorch-linknet)

[He, Zhang. Densely Connected Pyramid Dehazing Network](https://github.com/hezhangsprinter/DCPDN)

#### Install and Data

The code has been developed and tested with pytorch 1.0+ in a conda environment

```bash
conda activate {Project Environment}
git clone ${repository}
cd IRP
conda install --yes --file requirements.txt
cd ..
git clone https://github.com/Po-Hsun-Su/pytorch-ssim
git clone https://github.com/jorge-pessoa/pytorch-msssim
cd pytorch-ssim 
python setup.py
cd ../pytorch-msssim
python setup.py
```

Datasets used were provided by NITRE and the He, Zhang training Dataset found at https://github.com/hezhangsprinter/DCPDN

#### Training and Test
##### Training Airlight Model
```bash
# Train
python train.py configs/train_atmos.py
# Test
python test.py configs/test_atmos.py {airlight weights}
```
##### Training Transmission Model
```bash
# Train
python train.py configs/train_trans.py
# Test
python test.py configs/test_trans.py {trans weights}
``` 
##### Training Full Model on He Zhang Dataset
```bash
# Train
python train.py configs/train_full.py {trans weights} {airlight weights} 
# Test
python test.py configs/test_full.py {full weights}
``` 
##### Fine Tunning Full Model on NITRE Dataset
```bash
# Train
python train.py configs/train_nitre.py {full weights} 
# Test
python test.py configs/test_nitre.py {nitre weights}
``` 
#### Results
