# hdrnet-pytorch-2023
The repository contains a simple implementation of HDRNet (Deep Bilateral Learning for Real-Time Image Enhancement, SIGGRAPH 2017) in 2023.

python 3.8

pytorch 1.3.1

cuda 10.2

## description
Since I needed to do research on HDRNET's learnable bilateral grid, I reimplemented it with Pytorch.


Trilinear interpolation discards the edge area.


Without the implementation of the CUDA operator, the training will be very slow. I think I'll improve this in the future.

## train

### dataset
Crop the Fivek dataset image and import it into the folder dataset/train.

##### request
Images with a resolution of more than 1024 * 1024 -> dataset/train/full -> crop/flip/rotate -> input-full


Align with the upper left corner of Full Img -> dataset/train/gt -> crop/flip/rotate -> input-gt

### then
```sh
python3 train.py 
```
##### output
epoch_xxx.tar

## test

### dataset
Crop the Fivek dataset image and import it into the folder dataset/test/img.

##### request
1024 * 1024 -> dataset/test/img  

### then
```sh
python3 test.py
```
##### output
960 * 960 -> dataset/test/output


## code by
Wu F.Y 

## If
Modify the graphics resolution and network size : network :: FullNet & LowNet

Modify the channal : train & train

## about
@article{gharbi2017deep,

  title={Deep bilateral learning for real-time image enhancement},

  author={Gharbi, Micha{\"e}l and Chen, Jiawen and Barron, Jonathan T and Hasinoff, Samuel W and Durand, Fr{\'e}do},

  journal={ACM Transactions on Graphics (TOG)},

  volume={36},

  number={4},

  pages={118},

  year={2017},

  publisher={ACM}
  
}
