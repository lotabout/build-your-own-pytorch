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
    def __init__(self, x, stride=(1,1), padding=(0,0)):
        super(OpConv2d, self).__init__()
        self.x = x # (n, c, h, w)
        self.stride = stride # (h, w)
        if padding != (0,0):
            p_h, p_w = padding
            self.x = np.pad(x, ((0,0), (0,0), (p_h,p_h), (p_w, p_w)))

    def forward(self, weight: NDArray):
        # w is a 2d matrix
        i_n, i_c, i_h, i_w = self.x.shape
        s_h, s_w = self.stride
        w_h, w_w = weight.shape

        o_n = i_n
        o_c = 1
        o_h = (i_h - w_h) // s_h
        o_w = (i_w - w_w) // s_w

        out = np.zeros((o_n, o_c, o_h, o_w), dtype=weight.dtype)
        for n in range(i_n):
            for c in range(i_c):
                for h in range(o_h):
                    for w in range(o_w):
                        sel = self.x[:, :, h*s_h:h*s_h+w_h, w*s_w:w*s_w+w_w]
                        mul = sel * weight
                        out[:, 0, h, w] = np.sum(mul, axis=(1, 2, 3)) # sum over c

        return out

    def backward(self, grad: NDArray):
        ret = np.cos(self.x) * grad
        return ret


