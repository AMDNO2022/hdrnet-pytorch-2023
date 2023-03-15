# hdrnet-pytorch-2023
The repository contains a alternative implementation of HDRNet (Deep Bilateral Learning for Real-Time Image Enhancement, SIGGRAPH 2017) in 2023.

# This repository is not finished yet

python 3.8

pytorch 1.13.1

cuda 11.6

## description
Since I needed to study the usability of HDRNet for my research, I developed an HDRNet-like network. To develop efficiency, I used 1 * 1 convolutional layers instead of pixelwisenet and 16 * 16 deconvolutional layers instead of bilateral grid upsampling.

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
1024 * 1024 -> dataset/test/output


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
