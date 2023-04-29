from autograd import Tensor, Operator, NDArray
from ops import conv2d, avgPool2d, flatten, linear, sigmoid
import numpy as np


class Layer(object):
    def __init__(self):
        super(Layer, self).__init__()

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplemented("Subclass should implement")

    def update_gradient(self, updater):
        """Apply the gradient

        :param: updater: lambda(ndarray: NDArray, grad: NDArray): -> NDArray
        """
        raise NotImplemented("Subclass should implement")

class LazyConv2d(Layer):
    def __init__(self, out_channel, kernel_size, padding=0, stride=1):
        super(LazyConv2d, self).__init__()
        self.initialized = False # in channel unknown
        self.weight = None
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        if not self.initialized:
            shape = (self.out_channel, x.shape[1], self.kernel_size, self.kernel_size)
            self.weight = Tensor(np.random.randn(*shape), requires_grad=True)
            self.initialized = True
        return conv2d(x, self.weight, padding=self.padding, stride=self.stride)

    def update_gradient(self, updater):
        self.weight = Tensor(updater(self.weight.ndarray, self.weight.grad.ndarray), requires_grad=True)

class AvgPool2d(Layer):
    def __init__(self, kernel_size, stride):
        super(AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return avgPool2d(x, kernel_size=self.kernel_size, stride=self.stride)

    def update_gradient(self, updater):
        pass

class Flatten(Layer):
    def forward(self, x):
        return flatten(x)

    def update_gradient(self, updater):
        pass

class Sigmoid(Layer):
    def forward(self, x):
        return sigmoid(x)

    def update_gradient(self, updater):
        pass

class LazyLinear(Layer):
    def __init__(self, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.initialized = False
        self.out_features = out_features
        self.bias_enabled = True
        self.weight = None
        self.bias = None

    def forward(self, x):
        if not self.initialized:
            self.weight = Tensor(np.random.randn(self.out_features, x.shape[1]), requires_grad=True)
            self.bias = Tensor(np.random.randn(self.out_features), requires_grad=True) if self.bias_enabled is True else None
            self.initialized = True

        return linear(x, self.weight, self.bias)

    def update_gradient(self, updater):
        self.weight = Tensor(updater(self.weight.ndarray, self.weight.grad.ndarray), requires_grad=True)
        if self.bias is not None:
            self.bias= Tensor(updater(self.bias.ndarray, self.bias.grad.ndarray), requires_grad=True)

class Sequential(Layer):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = layers

    def forward(self, *args):
        input = args
        for layer in self.layers:
            if isinstance(input, tuple):
                input = layer.forward(*input)
            else:
                input = layer.forward(input)
        return input

    def update_gradient(self, updater):
        for layer in self.layers:
            layer.update_gradient(updater)
