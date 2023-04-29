from autograd import Tensor, Operator, NDArray
import unittest

import numpy as np

#======================================================================
def is_tensor(x):
    return isinstance(x, Tensor)

def is_scalar(x):
    # this implementation is not quite correct in corner cases
    # but works in our examples
    return not isinstance(x, Tensor)

def _pad(x: NDArray, padding):
    if padding != (0,0):
        p_h, p_w = padding
        return np.pad(x, ((0,0), (0,0), (p_h,p_h), (p_w, p_w)))
    return x

def _adjust_shape_with_sum(x: NDArray, out_shape: NDArray):
    if x.shape == out_shape:
        return x
    elif x.shape < out_shape:
        return np.broadcast_to(x, out_shape)
    else:
        compress_axis = [i for i, (x, y) in enumerate(zip(x.shape, out_shape)) if x != y]
        return np.sum(x, axis=tuple(compress_axis), keepdims=True)

#======================================================================
# Operator definition

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpExp(Operator):
    # func: y  = e^x
    # deri: y'/x' = e^x

    def forward(self, x: NDArray):
        self.x = x
        return np.exp(x)

    def backward(self, y: NDArray):
        return np.exp(self.x) * y

def exp(x):
    return OpExp()(x)

class TestOpExp(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        x = np.exp(np.random.randn(3, 5))
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad = True
        out = torch.exp(x_tensor)
        out.backward(2*torch.ones_like(out))

        op = OpExp()
        y = op.forward(x)
        x_prim = op.backward(2*np.ones_like(y))

        self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
        self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpLog(Operator):
    # func: y  = log(x)
    # deri: y'/x' = 1/x

    def forward(self, x: NDArray):
        self.x = x
        return np.log(x)

    def backward(self, y: NDArray):
        return y/self.x

def log(x):
    return OpLog()(x)

class TestOpLog(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        x = np.exp(np.random.randn(3, 5))
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad = True
        out = torch.log(x_tensor)
        out.backward(2*torch.ones_like(out))

        op = OpLog()
        y = op.forward(x)
        x_prim = op.backward(2*np.ones_like(y))

        self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
        self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpEWiseMul(Operator):
    # func: y = a * b
    # deri: y'/a' = b
    # deri: y'/b' = a
    def forward(self, a: NDArray, b: NDArray):
        self.a = a
        self.b = b
        return a * b

    def backward(self, grad: NDArray):
        return _adjust_shape_with_sum(self.b * grad, self.a.shape), \
               _adjust_shape_with_sum(self.a * grad, self.b.shape)

class TesetOpEWiseMul(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        shapes = [(1,1), (1,3), (3, 1), (3, 3)]
        for x_shape in shapes:
            for y_shape in shapes:
                x = np.random.randn(*x_shape)
                y = np.random.randn(*y_shape)
                x_tensor = torch.from_numpy(x)
                x_tensor.requires_grad = True
                y_tensor = torch.from_numpy(y)
                y_tensor.requires_grad = True
                t = x_tensor * y_tensor
                t.backward(2*torch.ones_like(t))
                op = OpEWiseMul()
                m = op.forward(x, y)
                x_prim, y_prim = op.backward(2*np.ones_like(m))
                self.assertTrue(np.all(t.detach().numpy() - m < 1e-6))
                self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))
                self.assertTrue(np.all(y_tensor.grad.detach().numpy() - y_prim < 1e-6))

class OpMulScalar(Operator):
    # func: y = a * scalar
    # deri: y'/a' = scalar
    def __init__(self, scalar):
        super(OpMulScalar, self).__init__()
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a * self.scalar

    def backward(self, grad: NDArray):
        return self.scalar * grad

class TesetOpMulScalar(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        x = np.random.randn(3, 5)
        x_tensor = torch.from_numpy(x)
        x_tensor.requires_grad = True
        t = x_tensor * 3
        t.backward(2*torch.ones_like(t))
        op = OpMulScalar(3)
        m = op.forward(x)
        x_prim = op.backward(2*np.ones_like(m))

        self.assertTrue(np.all(t.detach().numpy() - m < 1e-6))
        self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

def mul(a, b):
    if is_tensor(a) and is_tensor(b):
        return OpEWiseMul()(a, b)
    elif is_tensor(a) and is_scalar(b):
        return OpMulScalar(b)(a)
    elif is_scalar(a) and is_tensor(b):
        return OpMulScalar(a)(b)
    else:
        raise NotImplementedError(f'mul not implemented for two scalar')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpEWiseAdd(Operator):
    # func: y = a + b
    # deri: y'/a' = 1
    # deri: y'/b' = 1
    def forward(self, a: NDArray, b: NDArray):
        self.a = a
        self.b = b
        return a + b

    def backward(self, grad: NDArray):
        return _adjust_shape_with_sum(grad, self.a.shape), \
               _adjust_shape_with_sum(grad, self.b.shape)

class TesetOpEWiseAdd(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        shapes = [(1,1), (1,3), (3, 1), (3, 3)]
        for x_shape in shapes:
            for y_shape in shapes:
                x = np.random.randn(*x_shape)
                y = np.random.randn(*y_shape)
                x_tensor = torch.from_numpy(x)
                x_tensor.requires_grad = True
                y_tensor = torch.from_numpy(y)
                y_tensor.requires_grad = True
                t = x_tensor + y_tensor
                t.backward(2*torch.ones_like(t))
                op = OpEWiseAdd()
                m = op.forward(x, y)
                x_prim, y_prim = op.backward(2*np.ones_like(m))
                self.assertTrue(np.all(t.detach().numpy() - m < 1e-6))
                self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))
                self.assertTrue(np.all(y_tensor.grad.detach().numpy() - y_prim < 1e-6))

class OpAddScalar(Operator):
    # func: y = a + c
    # deri: y'/a' = 1

    def __init__(self, scalar):
        super(OpAddScalar, self).__init__()
        self.scalar = scalar

    def forward(self, a: NDArray):
        return self.scalar + a

    def backward(self, grad: NDArray):
        return grad

def add(a, b):
    if is_tensor(a) and is_tensor(b):
        return OpEWiseAdd()(a, b)
    elif is_tensor(a) and is_scalar(b):
        return OpAddScalar(b)(a)
    elif is_scalar(a) and is_tensor(b):
        return OpAddScalar(a)(b)
    else:
        raise NotImplementedError(f'add not implemented for two scalar')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpEWiseDiv(Operator):
    # func: y = a / b
    # deri: a' = 1/b
    # deri: b' = a * (-1/b^2)
    def forward(self, a: NDArray, b: NDArray):
        self.a = a
        self.b = b
        return a / b

    def backward(self, grad: NDArray):
        a_prim = 1/self.b * grad
        b_prim = self.a * (-1/(self.b ** 2)) * grad
        return _adjust_shape_with_sum(a_prim, self.a.shape), \
               _adjust_shape_with_sum(b_prim, self.b.shape)

def div(a, b):
    return OpEWiseDiv()(a, b)

class TestOpEWiseDiv(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        shapes = [(1,1), (1,3), (3, 1), (3, 3)]
        for x_shape in shapes:
            for y_shape in shapes:

                x = np.random.randn(*x_shape)
                y = np.random.randn(*y_shape)
                x_tensor = torch.from_numpy(x)
                x_tensor.requires_grad = True
                y_tensor = torch.from_numpy(y)
                y_tensor.requires_grad = True
                t = x_tensor / y_tensor
                t.backward(2*torch.ones_like(t))
                op = OpEWiseDiv()
                m = op.forward(x, y)
                x_prim, y_prim = op.backward(2*np.ones_like(m))

                self.assertTrue(np.all(t.detach().numpy() - m < 1e-6))
                self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))
                self.assertTrue(np.all(y_tensor.grad.detach().numpy() - y_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpSum(Operator):
    # func: y  = sum(x)
    # deri: y'/x' = broadcast

    def __init__(self, axis=None, keepdims=False):
        super(OpSum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: NDArray):
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, y: NDArray):
        return _adjust_shape_with_sum(y, self.x_shape)

def sum(x, axis=None, keepdims=False):
    return OpSum(axis=axis, keepdims=keepdims)(x)

class TestOpSum(unittest.TestCase):
    def test(self):
        import numpy as np
        import torch
        for axis in range(0,2):
            for keepdims in [True, False]:
                x = np.random.randn(5,5)
                x_tensor = torch.from_numpy(x)
                x_tensor.requires_grad = True
                out = torch.sum(x_tensor, axis=axis, keepdims=keepdims)
                loss = out.sum()
                loss.backward(2*torch.ones_like(loss))

                op = OpSum(axis=axis, keepdims=keepdims)
                y = op.forward(x)
                x_prim = op.backward(2*np.ones(out.shape))
                self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Square(Operator):
    # func: y = a + b
    # deri: y'/a' = 1
    # deri: y'/b' = 1
    def forward(self, a: NDArray):
        self.a = a
        return a ** 2

    def backward(self, grad: NDArray):
        ret = self.a * 2 * grad
        return ret

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpSin(Operator):
    # func: y = sin(x)
    # deri: y'/x' = cos(x)
    def forward(self, x: NDArray):
        self.x = x
        return np.sin(x)
    def backward(self, grad: NDArray):
        ret = np.cos(self.x) * grad
        return ret

def sin(x):
    return OpSin()(x)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpConv2d(Operator):
    def __init__(self, stride=(1,1), padding=(0,0)):
        super(OpConv2d, self).__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride # (h, w)
        self.padding = (padding, padding) if isinstance(padding, int) else padding # (h, w)

    @staticmethod
    def _conv2d_forward_w(x: NDArray, weight: NDArray, stride=(1,1), padding=(0,0)):
        # x is a 4d matrix (n, c, h, w)
        # weight is a 4d matrix (o_c, c, h, w), where o_c is output channel
        # out is (n, o_c, h, w)
        x_padded = _pad(x, padding)
        i_n, i_c, i_h, i_w = x_padded.shape
        s_h, s_w = stride
        wo_c, wi_c, w_h, w_w = weight.shape

        o_n = i_n
        o_c = wo_c
        o_h = (i_h - w_h) // s_h + 1
        o_w = (i_w - w_w) // s_w + 1

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=weight.dtype)
        for c in range(o_c):
            for h in range(o_h):
                for w in range(o_w):
                    sel = x_padded[:, :, h*s_h:h*s_h+w_h, w*s_w:w*s_w+w_w]
                    mul = sel * weight[c, :, :, :]
                    out[:, c, h, w] = np.sum(mul, axis=(1, 2, 3)) # sum over c
        return out

    @staticmethod
    def _conv2d_backward_y(x: NDArray, y: NDArray, stride=(1,1), padding=(0,0)):
        # x is a 4d matrix (n, c, h, w)
        # y is a 4d matrix (n, o_c, h, w)
        # output(w)     is (o_c, c, h, w)
        x_padded = _pad(x, padding)
        i_n, i_c, i_h, i_w = x_padded.shape
        s_h, s_w = stride
        y_n, yo_c, y_h, y_w = y.shape

        o_n = yo_c
        o_c = i_c
        o_h = (i_h - y_h) // s_h + 1
        o_w = (i_w - y_w) // s_w + 1

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=y.dtype)
        for c in range(o_n):
            for h in range(o_h):
                for w in range(o_w):
                    sel = x_padded[:, :, h*s_h:h*s_h+y_h, w*s_w:w*s_w+y_w]
                    mul = sel * y[:, c:c+1, :, :]
                    out[c, :, h, w] = np.sum(mul, axis=(0, 2, 3)) # sum over c
        return out

    @staticmethod
    def _conv2d_backward_w(y: NDArray, weight: NDArray, stride=(1,1), padding=(0,0)):
        # y is a 4d matrix (n, c, h, w)
        # w is a 4d matrix (c, o_c, h, w), where o_c is output channel
        # out y is         (n, o_c, h, w)
        y_padded = _pad(y, padding)
        i_n, i_c, i_h, i_w = y_padded.shape
        s_h, s_w = stride
        w_n, wo_c, w_h, w_w = weight.shape

        o_n = i_n
        o_c = wo_c
        o_h = (i_h - w_h) // s_h + 1
        o_w = (i_w - w_w) // s_w + 1

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=weight.dtype)
        for c in range(o_c):
            for h in range(o_h):
                for w in range(o_w):
                    sel = y_padded[:, :, h*s_h:h*s_h+w_h, w*s_w:w*s_w+w_w]
                    mul = sel * weight[:, c, :, :]
                    out[:, c, h, w] = np.sum(mul, axis=(1, 2, 3)) # sum over c
        return out

    def forward(self, x: NDArray, weight: NDArray):
        self.x_padded = _pad(x, self.padding)
        if len(weight.shape) == 3:
            self.weight = weight.reshape((1, *weight.shape))
        else:
            self.weight = weight
        return OpConv2d._conv2d_forward_w(self.x_padded, weight, self.stride)

    def backward(self, grad: NDArray):
        # grad: (n, h, c, w)
        # y = conv2d(x, w)
        # w' = conv2d(x, y')
        # x' = conv2d(flip(w), y')

        x_n, x_c, x_h, x_w = self.x_padded.shape
        w_n, w_c, w_h, w_w = self.weight.shape
        y_n, y_c, y_h, y_w = grad.shape
        s_h, s_w = self.stride

        if self.stride == (1,1):
            y_expanded = grad
        else:
            # [[1,2], [3,4]] => [[1,0,2], [0,0,0], [3,0,4]]
            h = x_h - w_h + 1
            w = x_w - w_w + 1
            y_expanded = np.zeros((y_n, y_c, h, w))
            for h in range(y_h):
                for w in range(y_w):
                    y_expanded[:, :, h*s_h, w*s_w] = grad[:,:,h, w]

        p_h, p_w = self.padding
        _, _, w_h, w_w = self.weight.shape

        w_prim = OpConv2d._conv2d_backward_y(self.x_padded, y_expanded)
        w_flip = np.flip(self.weight, (2,3))
        x_padded_prim = OpConv2d._conv2d_backward_w(y_expanded, w_flip, padding=(w_h-1, w_w-1))

        _, _, x_h, x_w = x_padded_prim.shape

        # remove padding
        if self.padding != (0,0):
            x_prim = x_padded_prim[:, :, p_h:x_h-p_h, p_w:x_w-p_w]
        else:
            x_prim = x_padded_prim

        return x_prim, w_prim

def conv2d(x, w, stride=(1,1), padding=(0,0)):
    return OpConv2d(stride=stride, padding=padding)(x, w)

class TestOpConv2d(unittest.TestCase):

    def test_OpConv2d(self):
        import numpy as np
        import torch

        for k in range(1,6):
            for s in range(1,4):
                for p in range(0,k//2):
                    for in_channels in range(1,4):
                        for out_channel in range(1,4):
                            for hw in range(k+3, k*2+3):
                                with self.subTest(kernel_size=k, stride=s, padding=p, in_channels=in_channels, out_channels=out_channel, hw=hw):
                                    conv = torch.nn.Conv2d(
                                        in_channels=in_channels,
                                        out_channels=out_channel,
                                        kernel_size=k,
                                        bias=False,
                                        stride = s,
                                        padding_mode='zeros',
                                        padding=p
                                    )

                                    x = np.random.randn(4,in_channels,hw,hw)
                                    w = np.random.randn(out_channel,in_channels,k,k)

                                    x_tensor = torch.from_numpy(x)
                                    x_tensor.requires_grad = True
                                    conv.weight = torch.nn.Parameter(torch.from_numpy(w))
                                    out = conv(x_tensor)
                                    loss = out.sum()
                                    loss.backward(2*torch.ones_like(loss))

                                    op = OpConv2d(padding=(p,p), stride=(s,s))
                                    y = op.forward(x, w)
                                    x_prim, w_prim = op.backward(2*np.ones(out.shape))

                                    self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                                    self.assertTrue(np.all(conv.weight.grad.detach().numpy() - w_prim < 1e-6))
                                    self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpSigmoid(Operator):
    # all element wise
    # func: y = 1/(1+e^(-x))
    # deri: y' = y * (1-y)

    def forward(self, x: NDArray):
        self.y = 1/(1+np.exp(-x))
        return self.y

    def backward(self, grad: NDArray):
        return grad * self.y * (1-self.y)

def sigmoid(x):
    return OpSigmoid()(x)

class TestOpSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        import numpy as np
        import torch

        for n in range(1,4):
            for c in range(1,4):
                for hw in range(3, 6):
                    sigmoid = torch.nn.Sigmoid()
                    x = np.random.randn(n,c,hw,hw)
                    x_tensor = torch.from_numpy(x)
                    x_tensor.requires_grad = True
                    out = sigmoid(x_tensor)

                    loss = out.sum()
                    loss.backward(2*torch.ones_like(loss))

                    op = OpSigmoid()
                    y = op.forward(x)
                    x_prim = op.backward(2*np.ones(out.shape))

                    self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                    self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpAvgPool2d(Operator):
    def __init__(self, kernel_size, stride=(1,1), padding=(0,0)):
        super(OpAvgPool2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size # (h, w)
        self.stride = (stride, stride) if isinstance(stride, int) else stride # (h, w)
        self.padding = (padding, padding) if isinstance(padding, int) else padding # (h, w)

    def forward(self, x: NDArray):
        x_padded = _pad(x, self.padding)
        self.x_padded_shape = x_padded.shape
        x_n, x_c, x_h, x_w = x_padded.shape
        s_h, s_w = self.stride
        k_h, k_w = self.kernel_size

        o_n = x_n
        o_c = x_c
        o_h = (x_h - k_h) // s_h + 1
        o_w = (x_w - k_w) // s_w + 1

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=x.dtype)
        for h in range(o_h):
            for w in range(o_w):
                sel = x_padded[:, :, h*s_h:h*s_h+k_h, w*s_w:w*s_w+k_w]
                out[:, :, h, w] = np.average(sel, axis=(2, 3)) # average hxw
        return out

    def backward(self, grad: NDArray):
        x_n, x_c, x_h, x_w = self.x_padded_shape
        y_n, y_c, y_h, y_w = grad.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding

        if self.stride == (1,1):
            y_expanded = grad
        else:
            # [[1,2], [3,4]] => [[1,0,2], [0,0,0], [3,0,4]]
            h = x_h - k_h + 1
            w = x_w - k_w + 1
            y_expanded = np.zeros((y_n, y_c, h, w))
            for h in range(y_h):
                for w in range(y_w):
                    y_expanded[:, :, h*s_h, w*s_w] = grad[:,:,h, w]

        y_padded = _pad(y_expanded, (k_h-1, k_w-1))

        o_h = x_h - p_h
        o_w = x_w - p_w
        out = np.zeros((x_n, x_c, o_h, o_w), dtype=y_padded.dtype)
        for h in range(o_h):
            for w in range(o_w):
                sel = y_padded[:, :, h:h+k_h, w:w+k_w]
                out[:, :, h, w] = np.sum(sel, axis=(2, 3))/4

        # remove padding
        if self.padding != (0,0):
            return out[:, :, p_h:x_h-p_h, p_w:x_w-p_w]
        return out

def avgPool2d(x, kernel_size, stride=(1,1), padding=(0,0)):
    return OpAvgPool2d(kernel_size, stride=stride, padding=padding)(x)

class TestOpAvgPool2d(unittest.TestCase):

    def test_OpAvgPool2d(self):
        import numpy as np
        import torch

        for k in range(1,6):
            for s in range(1,4):
                for p in range(0,k//2):
                    for hw in range(k+3, k*2+3):
                        with self.subTest(kernel_size=k, stride=s, padding=p, hw=hw):
                            avgPool = torch.nn.AvgPool2d(
                                kernel_size=k,
                                stride=s,
                                padding=p
                            )

                            x = np.random.randn(5,5,hw,hw)
                            x_tensor = torch.from_numpy(x)
                            x_tensor.requires_grad = True
                            out = avgPool(x_tensor)

                            loss = out.sum()
                            loss.backward(2*torch.ones_like(loss))

                            op = OpAvgPool2d(kernel_size=(avgPool.kernel_size,avgPool.kernel_size),
                                           padding=(avgPool.padding,avgPool.padding),
                                           stride=(avgPool.stride,avgPool.stride))
                            y = op.forward(x)
                            x_prim = op.backward(2*np.ones(out.shape))

                            self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                            self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpReLu(Operator):

    def forward(self, x: NDArray):
        return np.maximum(0, x)

    def backward(self, grad: NDArray):
        return np.maximum(0, grad)

def relu(x):
    return OpReLu()(x)

class TestOpReLu(unittest.TestCase):

    def test_relu(self):
        import numpy as np
        import torch

        for n in range(1,4):
            for c in range(1,4):
                for hw in range(3, 6):
                    relu = torch.nn.ReLU()
                    x = np.random.randn(n,c,hw,hw)
                    x_tensor = torch.from_numpy(x)
                    x_tensor.requires_grad = True
                    out = relu(x_tensor)

                    loss = out.sum()
                    loss.backward(2*torch.ones_like(loss))

                    op = OpReLu()
                    y = op.forward(x)
                    x_prim = op.backward(2*np.ones(out.shape))

                    self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                    self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpFlatten(Operator):
    def __init__(self, start_dim=1, end_dim=-1):
        super(OpFlatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: NDArray):
        self.x_shape = x.shape
        end = self.end_dim % len(self.x_shape) + 1
        merged = np.prod(self.x_shape[self.start_dim: end])
        out_shape = self.x_shape[0:self.start_dim] + (merged,) + self.x_shape[end:]

        return x.reshape(out_shape)

    def backward(self, grad: NDArray):
        return grad.reshape(self.x_shape)
        
def flatten(x):
    return OpFlatten()(x)

class TestOpFltten(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        for n in range(1,4):
            for c in range(1,4):
                for hw in range(3, 6):
                    flatten = torch.nn.Flatten()
                    x = np.random.randn(n,c,hw,hw)
                    x_tensor = torch.from_numpy(x)
                    x_tensor.requires_grad = True
                    out = flatten(x_tensor)

                    loss = out.sum()
                    loss.backward(2*torch.ones_like(loss))

                    op = OpFlatten()
                    y = op.forward(x)
                    x_prim = op.backward(2*np.ones(out.shape))

                    self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                    self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpLinear(Operator):

    def forward(self, x: NDArray, w: NDArray, b: NDArray):
        # x: (n, id) n samples, in_d dimensions
        # w: (od, id) in_d dimensions, out_d dimensions
        # b: (od) out_d dimensions
        # y: (n, od) = x * w.T + b
        self.x = x
        self.w = w
        self.b = b

        if self.b is None:
            return np.matmul(x, w.T)
        else:
            return np.matmul(x, w.T) + b

    def backward(self, grad):
        # func: y = x * w.T + b
        # deri: y'/x' = y' * w
        # deri: y'/w' = y'.T * x
        # deri: y'/b' = y'

        if self.b is None:
            return np.matmul(grad, self.w), np.matmul(grad.T, self.x)
        else:
            return np.matmul(grad, self.w), np.matmul(grad.T, self.x), np.sum(grad, axis=0, keepdims=True)

def linear(x, w, b=None):
    return OpLinear()(x, w, b)

class TestOpLinear(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        for n in range(1,4):
            for id in range(10,15):
                for od in range(10, 15):
                    linear = torch.nn.Linear(id, od)
                    x = np.random.randn(n, id)
                    x_tensor = torch.from_numpy(x)
                    x_tensor.requires_grad = True
                    w = np.random.randn(od, id)
                    linear.weight = torch.nn.Parameter(torch.from_numpy(w))
                    b = np.random.randn(od)
                    linear.bias = torch.nn.Parameter(torch.from_numpy(b))

                    out = linear(x_tensor)

                    loss = out.sum()
                    loss.backward(2*torch.ones_like(loss))

                    op = OpLinear()
                    y = op.forward(x, w, b)
                    x_prim, w_prim, b_prim = op.backward(2*np.ones(out.shape))

                    self.assertTrue(np.all(out.detach().numpy() - y < 1e-6))
                    self.assertTrue(np.all(x_tensor.grad.detach().numpy() - x_prim < 1e-6))
                    self.assertTrue(np.all(linear.weight.grad.detach().numpy() - w_prim < 1e-6))
                    self.assertTrue(np.all(linear.bias.grad.detach().numpy() - b_prim < 1e-6))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpCrossEntropy(Operator):
    def __init__(self, reduction='mean'):
        super(OpCrossEntropy, self).__init__()
        self.reduction = reduction
        if self.reduction not in ['mean', 'sum']:
            raise Exception('reduction could only be [mean | sum]')

    def forward(self, x: NDArray, y: NDArray):
        # x: (mini-batches, classes)
        # y: (mini-batches, classes)
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        self.x_tensor = Tensor(x, requires_grad=True)
        self.y_tensor = Tensor(y, requires_grad=True)

        x_exp = exp(self.x_tensor)
        x_sum = sum(x_exp, axis=1, keepdims=True)
        x_softmax = div(x_exp, x_sum)
        x_log = log(x_softmax)
        self.sum = mul(-1, sum(mul(x_log, self.y_tensor)))

        if self.reduction == 'mean':
            return self.sum.ndarray / x.shape[0]
        elif self.reduction == 'sum':
            return self.sum.ndarray
        else:
            raise Exception('reduction could only be [mean | sum]')

    def backward(self, grad: NDArray):
        # grad: (1,)
        if self.reduction == 'mean':
            grad = grad / self.x_tensor.shape[0]

        self.sum.backward(grad)
        return self.x_tensor.grad.ndarray, self.y_tensor.grad.ndarray

class TestOpCrossEntropy(unittest.TestCase):

    def test(self):
        import numpy as np
        import torch

        shapes = [(1,1), (1,3), (3, 1), (3, 3)]
        for shape in shapes:
            for reduction in ['sum', 'mean']:

                loss = torch.nn.CrossEntropyLoss(reduction=reduction)
                x = np.random.randn(*shape)
                y = np.random.randn(*shape)
                x_tensor = torch.from_numpy(x)
                x_tensor.requires_grad = True
                y_tensor = torch.from_numpy(y)
                y_tensor.requires_grad = True
                out = loss(x_tensor, y_tensor)
                out.backward()
                op = OpCrossEntropy(reduction=reduction)
                t = op.forward(x, y)
                x_prim, y_prim = op.backward(np.ones(out.shape))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Tests
# invoke by python3 -m unittest ops

if __name__ == '__main__':
    unittest.main()
