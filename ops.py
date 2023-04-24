from autograd import Tensor, Operator, NDArray

import numpy as np

#======================================================================
def is_tensor(x):
    return isinstance(x, Tensor)

def is_scalar(x):
    # this implementation is not quite correct in corner cases
    # but works in our examples
    return not isinstance(x, Tensor)

#======================================================================
# Operator definition

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OpExp(Operator):
    # func: y  = e^x
    # deri: y'/x' = e^x

    def forward(self, x: NDArray):
        return np.exp(x)

    @staticmethod
    def backward(self, y: NDArray):
        return np.exp(x)

def exp(x):
    return OpExp()(x)

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
        return self.b * grad, self.a * grad

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
        return a + b

    def backward(self, grad: NDArray):
        ret = grad, grad
        return ret

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
        self.stride = stride # (h, w)
        self.padding = padding

    @staticmethod
    def _conv2d(x: NDArray, weight: NDArray, stride=(1,1), padding=(0,0)):
        # x is a 4d matrix (n, c, h, w)
        # weight is a 4d matrix (n, c, h, w), where w_n is 1 or equals to x_n
        if padding != (0,0):
            p_h, p_w = padding
            x_padded = np.pad(x, ((0,0), (0,0), (p_h,p_h), (p_w, p_w)))
        else:
            x_padded = x

        i_n, i_c, i_h, i_w = x_padded.shape
        s_h, s_w = stride
        if len(weight.shape) == 3:
            weight = weight.reshape((1, *weight.shape))
        w_n, w_c, w_h, w_w = weight.shape

        o_n = i_n
        o_c = w_c
        o_h = (i_h - w_h + 1) // s_h
        o_w = (i_w - w_w + 1) // s_w

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=weight.dtype)
        for c in range(o_c):
            for h in range(o_h):
                for w in range(o_w):
                    sel = x_padded[:, :, h*s_h:h*s_h+w_h, w*s_w:w*s_w+w_w]
                    mul = sel * weight[:, c, :, :]
                    out[:, c, h, w] = np.sum(mul, axis=(1, 2, 3)) # sum over c
        return out

    def forward(self, x: NDArray, weight: NDArray):
        if self.padding != (0,0):
            p_h, p_w = self.padding
            x_padded = np.pad(x, ((0,0), (0,0), (p_h,p_h), (p_w, p_w)))
        else:
            x_padded = x
        self.x_padded = x_padded
        self.weight = weight
        return OpConv2d._conv2d(x_padded, weight, self.stride)

    def backward(self, grad: NDArray):
        # grad: (n, h, c, w)
        # y = conv2d(x, w)
        # w' = conv2d(x, y')
        # x' = conv2d(flip(w), y')

        y_n, y_c, y_h, y_w = grad.shape
        s_h, s_w = self.stride

        if self.stride == (1,1):
            y_expanded = grad
        else:
            # [[1,2], [3,4]] => [[1,0,2], [0,0,0], [3,0,4]]
            h = y_h+(y_h-1)*(s_h-1)
            w = y_w+(y_w-1)*(s_w-1)
            y_expanded = np.zeros((y_n, y_c, h, w))
            for h in range(y_h):
                for w in range(y_w):
                    y_expanded[:, :, h*s_h, w*s_w] = grad[:,:,h, w]

        p_h, p_w = self.padding
        _, w_h, w_w = self.weight.shape

        w_prim = OpConv2d._conv2d(self.x_padded, y_expanded)
        w_flip = np.flip(self.weight, (1,2))
        x_padded_prim = OpConv2d._conv2d(y_expanded, w_flip, padding=(w_h//2, w_w//2))

        _, _, x_h, x_w = x_padded_prim.shape

        # remove padding
        if self.padding != (0,0):
            x_prim = x_padded_prim[:, :, p_h:x_h-p_h, p_w:x_w-p_w]
        else:
            x_prim = x_padded_prim

        return x_prim, w_prim
