#!/usr/bin/env python3

# High Level
# Tensor 需要知道自己的 Op 是哪来的吗？感觉不需要？

from typing import List, Optional, NamedTuple, Tuple, Union, Dict
import numpy as np
import collections

NDArray = np.ndarray

class Tensor(object):
    """tensor"""
    def __init__(self, ndarray: NDArray, requires_grad=False, grad_fn=None):
        super(Tensor, self).__init__()
        self.ndarray = ndarray
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.grad = None
        self._grad_accmulator = None

    def is_leaf(self) -> bool:
        return self.requires_grad and self.grad_fn is None

    def backward(self, output_grad):
        if self.grad_fn is None:
            raise "backward could not be called if grad_fn is None"
        execute_graph(self.grad_fn, output_grad)

    def __str__(self):
        grad_info = f' grad_fn={self.grad_fn}' if self.grad_fn is not None else ''
        return f'tensor({self.ndarray}{grad_info})'

    def __repr__(self):
        return self.__str__()

# 注意 Operator 里计算的都是 Tensor 内部的数据
class Operator(object):
    def __init__(self):
        super(Operator, self).__init__()
        self.next_ops = []

    def forward(self, *args: Tuple[NDArray]) -> NDArray:
        raise NotImplementedError("Should be override by subclass")

    def backward(self, output_grad: Tuple[NDArray]) -> Union[NDArray, Tuple[NDArray]]:
        raise NotImplementedError("Should be override by subclass")

    def __call__(self, *args: Tuple[Tensor]) -> Tensor:
        grad_fn = None
        requires_grad = any((t.requires_grad for t in args))

        if requires_grad:
            # add edges
            for input in args:
                if input.is_leaf():
                    if input._grad_accmulator is None:
                        input._grad_accmulator = OpAccumulate(input)
                    self.next_ops.append(input._grad_accmulator)
                else:
                    self.next_ops.append(input.grad_fn)
            grad_fn = self

        inputs = [t.ndarray for t in args]
        output = self.forward(*inputs)
        return Tensor(output, requires_grad=requires_grad, grad_fn=grad_fn)

class OpAccumulate(Operator):
    def __init__(self, tensor):
        super(OpAccumulate, self).__init__()
        self.tensor = tensor
    def backward(self, grad):
        self.tensor.grad = Tensor(grad)
        return grad

#======================================================================
# execute backward

from collections import deque
def compute_dependencies(root):
    # deps: {op: num}
    deps = {}
    q = deque()
    traversed = {root}
    q.append(root)
    while len(q) != 0:
        cur = q.pop()
        if len(cur.next_ops) == 0:
            continue
        for next in cur.next_ops:
            deps[next] = deps.get(next, 0) + 1
            if next not in traversed:
                q.append(next)
                traversed.add(next)
    return deps

def execute_graph(root, output_grad):
    deps = compute_dependencies(root)
    inputs = {root: output_grad}

    q = deque()
    q.append(root)
    while len(q) != 0:
        task = q.pop()
        input = inputs[task]
        outputs = task.backward(input)
        if not isinstance(outputs, collections.abc.Sequence):
            outputs = [outputs]

        for next_op, output in zip(task.next_ops, outputs):
            if next_op is None:
                continue

            # accumulate the "inputs" for next_op
            op_input = inputs.get(next_op, 0)
            inputs[next_op] = op_input + output

            deps[next_op] -= 1
            if deps[next_op] == 0:
                q.append(next_op)
