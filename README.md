# LipGrow
An adaptive training algorithm for residual network based on model Lipschitz

## Install
* Install [PyTorch](http://pytorch.org/)
* Clone recursively
  ```
  git clone --recursive https://github.com/shwinshaker/LipGrow.git
  ```

## Setup
* By default, build a `./data` directory including the datasets 
* By default, build a `./checkpoints` directory to save the training output

## Training
* CIFAR-10/100
  ```
  ./launch.sh
  ```
* Tiny-ImageNet
  ```
  ./imagenet-launch.sh
  ```

## Citation

If you find our algorithm helpful, consider citing our paper
> [Towards Adaptive Residual Network Training: A Neural-ODE Perspective](https://proceedings.icml.cc/static/paper_files/icml/2020/6462-Paper.pdf)

```
@inproceedings{Dong2020TowardsAR,
  title={Towards Adaptive Residual Network Training: A Neural-ODE Perspective},
  author={Chengyu Dong and Liyuan Liu and Zichao Li and Jingbo Shang},
  year={2020}
}
```