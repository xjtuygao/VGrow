# VGrow in PyTorch
PyTorch implementation of [Deep Generative Learning via Variational Gradient Flow](https://arxiv.org/abs/1901.08469).  

# Prerequisites
Python 3.5+  
PyTorch v0.4.1  

# Usage
To run VGrow on [MNIST](http://yann.lecun.com/exdb/mnist/), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) 
and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), use the following `cmd` with default arguments  
`python main.py --divergence KL --dataset <data_set> --dataroot <data_root>`
