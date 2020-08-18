# Manifold-attack
This is an implementation of manifold attack.
Khanh-Hung TRAN, Fred-Maurice NGOLE MBOULA, Jean-Luc STARCK

Require : pytorch


1, projections.py contains the projection for sum and positive constraint.

2, train_ImageNet.py is to train ImageNet dataset with 4 methods ERM, MixUp, CutMix and MixUpAttack. User need to fill paths for train and valid data folder. We use implementation of MixUp by Zhang "https://github.com/hongyi-zhang/mixup" and CutMix by Yun "https://github.com/clovaai/CutMix-PyTorch". An exemple of uisng this script : 

python train_ImageNet.py --method='MixUpAttack' --epochs=300 --out='/home/result' --manualSeed=0 --gpu_device 0 1 2 3 --batch-size=200 --beta=0.2 -j 16 --n_iters=1 --xi=0.01 --xi-end=0.01 --lock=False

Error rate :
| Method \ Data |   Top-1    |     Top-5   |
| ------------- | ------------- | -------------|
|ERM| 33.84 | 12.46 |
|MixUp| 32.13 | 11.35 |
|MixUpAttack| 31.35 | 10.81 |
|CutMix| 30.94 | 10.41 |

P/s : To download ImageNet dataset, we use the downloader written by Martins Frolovs "https://github.com/mf1024/ImageNet-Datasets-Downloader"

3, FGSM_ImageNet.py is to evaluate trained model by adversarial examples. User need to fill paths for trained models ERM, MixUp, CutMix and MixUpAttack

To run :

python FGSM_ImageNet.py

Error rate :
| Method \ Data |   Top-1    |     Top-5   |
| ------------- | ------------- | -------------|
|MixUp|76.79 | 50.76 |
|MixUpAttack| 71.31 | 44.5 |
|CutMix| 81.24 | 58.72 |

4, train_MixMatchAttack.py is to train on CIFAR-10 or SVHN dataset with MixMatchAttack method, an extension of MixMatch "https://arxiv.org/abs/1905.02249". We developpe the attack version from an implemention of MixMatch in Pytorch, written by Yui : "https://github.com/YU1ut/MixMatch-pytorch". An exemple of uisng this script :

python train_MixMatchAttack.py --data='cifar10' --n-labeled=250 --manualSeed=0 --out='/home/result' --gpu='0' --xi=0.1 --xi_end=0.01 --n_iters=1


Error rate in case of using 250 labelled samples :
| Data | Method \ Test |  1    |    2   |  3   |  4   | Mean |
|------|--------|--------|-------|-------|-------|-------|
|CIFAR-10| MixMatch|10.62|12.72|12.02|15.26|12.65±1.68|
|CIFAR-10| MixMatchAttack|8.84|10.46|10.09|12.89|10.57±1.47|
|SVHN| MixMatch|6.09|6.73|7.80|7.37|7.0±0.65|
|SVHN| MixMatchAttack|5.07|5.93|5.42|5.47|5.47±0.3|

Test 1,2,3,4 correpond to manualSeed = 0,1,2,3


