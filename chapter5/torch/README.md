

# Implementation for binary neural network on Torch

Whose aim of this implementation is replicating [binary neural network] (https://github.com/itayhubara/BinaryNet).

## Description

The implementation is heavily relyed on multi-dimensional tensor, but not based on an automatic differenciation function provided by torch.


The reason is following. 

Most Papers for binary neural networks represent algorithm overview with 3 steps; namely, feed-forward, feed-backward,and update-weights. However, automatic differenciation function of each libraries will not be able to seperate backward-process and update-weights-process, which makes difficult to be along with the description of papers.

Whether a computational graph is introduced or not should not be matter in terms of computational speed meaing if algorithm can be on GPU. 

## Usage

You can train for mnist with

`th mnistOnly.lua`

You can train cifar10 after you convert data to torch tensor executing

`cd ../data/cifar/cifar10`

`bash exe.sh`

since generated tensor is more than 100MB, it cannot be on github.

Then, train with

`th generalTrain.lua`

You can set a model as a table given by torch where you can create your own network architecture.

There are different data, MNIST,Cifar10,Cifar100,and SVHN which needs to be downloaded on /load/data folder.

you can examine what is a command line argument typing ".lua --help"

Iteration of each code is charted on the chart/chart1.svg as a default setting.

`display chart/chart1.svg`

