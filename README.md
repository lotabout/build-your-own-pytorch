# Build Your Own PyTorch

Understand deep learning framework(like torch) by implementing the essentials.
Start with Numpy only and ends with LeNet on MNIST dataset.

What's included?
- Auto gradient, so that we don't need to manually calculate partitial derivatives.
- Builtin Operators, e.g. (Conv2d, AvePool, CrossEntropy, etc.). Builtin
    operators are necessary for convenience and efficiency.
- Network Layers(`nn` module in torch) that wraps parameters and operators to
    make crafting network easier.
- LeNet, an implementation of LeNet that combines all.

## Quick start

```sh
# Download MNIST dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# move dataset into data directory
mkdir -p data
mv *.gz data

# install deps
pip install numpy
pip install python-mnist

# run lenet
python lenet.py
```

## Some thoughts

- autograd is relatively easy
- tensors are hard, especially considering mini-batch and broadcasting...

## Benchmark of ops?

```
OpConv2d-backward        : avg: 0.0953309418, counts: 81
OpConv2d-forward         : avg: 0.0331299218, counts: 82
OpAvgPool2d-backward     : avg: 0.0114380703, counts: 82
OpSigmoid-forward        : avg: 0.0023136401, counts: 164
OpAvgPool2d-forward      : avg: 0.0016568696, counts: 82
OpSigmoid-backward       : avg: 0.0008361994, counts: 164
OpLinear-backward        : avg: 0.0003190138, counts: 123
OpCrossEntropy-backward  : avg: 0.0002894227, counts: 41
OpCrossEntropy-forward   : avg: 0.0001843266, counts: 41
OpLinear-forward         : avg: 0.0001273892, counts: 123
OpEWiseDiv-backward      : avg: 0.0000668037, counts: 41
OpFlatten-forward        : avg: 0.0000345940, counts: 41
OpSum-backward           : avg: 0.0000244466, counts: 82
OpSum-forward            : avg: 0.0000213559, counts: 82
OpEWiseMul-backward      : avg: 0.0000208762, counts: 41
OpExp-backward           : avg: 0.0000152937, counts: 41
OpLog-forward            : avg: 0.0000133398, counts: 41
OpExp-forward            : avg: 0.0000130607, counts: 41
OpEWiseDiv-forward       : avg: 0.0000130432, counts: 41
OpEWiseMul-forward       : avg: 0.0000070130, counts: 41
OpLog-backward           : avg: 0.0000053208, counts: 41
OpFlatten-backward       : avg: 0.0000038147, counts: 41
OpMulScalar-forward      : avg: 0.0000034832, counts: 41
OpMulScalar-backward     : avg: 0.0000019829, counts: 41
```
