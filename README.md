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
python lenet.py
```

Note that current implementation is slow. And main bottleneck is the naive
implementation of `Conv2d` operator.

## Some thoughts

- autograd is relatively easy
- tensors are hard, especially considering mini-batch and broadcasting...

## Benchmark of ops?

```
OpConv2d-backward        : avg: 0.2599260330, counts: 40
OpConv2d-forward         : avg: 0.1358403578, counts: 41
OpAvgPool2d-backward     : avg: 0.0219547331, counts: 40
OpAvgPool2d-forward      : avg: 0.0060435712, counts: 40
OpSigmoid-forward        : avg: 0.0020532936, counts: 80
OpSigmoid-backward       : avg: 0.0013529807, counts: 80
OpLinear-backward        : avg: 0.0003633142, counts: 60
OpCrossEntropy-forward   : avg: 0.0003455400, counts: 20
OpCrossEntropy-backward  : avg: 0.0002197146, counts: 20
OpLinear-forward         : avg: 0.0002145807, counts: 60
OpEWiseDiv-backward      : avg: 0.0000467539, counts: 20
OpFlatten-forward        : avg: 0.0000318766, counts: 20
OpSum-backward           : avg: 0.0000212908, counts: 40
OpSum-forward            : avg: 0.0000176311, counts: 40
OpExp-backward           : avg: 0.0000113964, counts: 20
OpLog-forward            : avg: 0.0000101089, counts: 20
OpEWiseDiv-forward       : avg: 0.0000095367, counts: 20
OpExp-forward            : avg: 0.0000094056, counts: 20
OpEWiseMul-backward      : avg: 0.0000084162, counts: 20
OpEWiseMul-forward       : avg: 0.0000053644, counts: 20
OpLog-backward           : avg: 0.0000038862, counts: 20
OpFlatten-backward       : avg: 0.0000031948, counts: 20
OpMulScalar-forward      : avg: 0.0000014186, counts: 20
OpMulScalar-backward     : avg: 0.0000013828, counts: 20
```
