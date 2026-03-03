import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from mini_torch.tensor import Tensor
from mini_torch.nn.layers import Linear


EPS = 1e-3
ATOL = 2e-2
RTOL = 2e-2

# ---------------------------------------------Helper functions for testing---------------------------------------------
def _np(t: Tensor) -> np.ndarray:
    if hasattr(t, "numpy"):
        return np.array(t.detach().numpy(), dtype=np.float32)
    return np.array(t.data, dtype=np.float32)


def _finite_diff_grad_param(f, param_np, eps=EPS):
    grad = np.zeros_like(param_np, dtype=np.float32)
    it = np.nditer(param_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(param_np[idx])

        param_np[idx] = old + eps
        fp = f(param_np)

        param_np[idx] = old - eps
        fm = f(param_np)

        param_np[idx] = old
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad

# Automatically identify Linear's weight and bias from layer.parameters() by matching shapes, without relying on attribute names.
def _get_linear_params_by_shape(layer, Din, Dout, expect_bias=True):
    params = list(layer.parameters())

    weight = None
    bias = None

    for p in params:
        shape = tuple(p.data.shape)
        if shape == (Din, Dout):
            if weight is not None:
                raise AssertionError(f"Found multiple candidate weight parameters with shape {(Din, Dout)}.")
            weight = p
        elif shape == (Dout,):
            if bias is not None:
                raise AssertionError(f"Found multiple candidate bias parameters with shape {(Dout,)}.")
            bias = p

    if weight is None:
        raise AssertionError(f"Could not find Linear weight parameter with shape {(Din, Dout)}.")

    if expect_bias and bias is None:
        raise AssertionError(f"Could not find Linear bias parameter with shape {(Dout,)}.")

    return weight, bias


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)

#forward shape and value test
def test_linear_forward_1():
    """
    Forward test (with bias):
      - Manually set W and b
      - Compare layer(x) against NumPy reference: x @ W + b
      - Also checks output shape is (N, Dout)
    """
    N, Din, Dout = 4, 5, 3
    x0 = np.random.randn(N, Din).astype(np.float32)

    layer = Linear(Din, Dout, bias=True)
    W_param, b_param = _get_linear_params_by_shape(layer, Din, Dout, expect_bias=True)

    W0 = np.random.randn(Din, Dout).astype(np.float32)
    b0 = np.random.randn(Dout).astype(np.float32)
    W_param.data[...] = W0
    b_param.data[...] = b0

    x = Tensor(x0.copy(), requires_grad=False)
    y = layer(x)

    y_np = _np(y)
    ref = x0 @ W0 + b0

    assert y_np.shape == (N, Dout)
    np.testing.assert_allclose(y_np, ref, rtol=1e-6, atol=1e-6)


#test when bias==False
def test_linear_forward_2():
    """
    Forward test (bias=False):
      - Manually set W
      - Compare layer(x) against NumPy reference: x @ W
      - Ensures bias is not added and output shape matches
    """
    N, Din, Dout = 2, 4, 6
    x0 = np.random.randn(N, Din).astype(np.float32)

    layer = Linear(Din, Dout, bias=False)
    W_param, _ = _get_linear_params_by_shape(layer, Din, Dout, expect_bias=False)

    W0 = np.random.randn(Din, Dout).astype(np.float32)
    W_param.data[...] = W0

    x = Tensor(x0.copy(), requires_grad=False)
    y = layer(x)

    y_np = _np(y)
    ref = x0 @ W0

    assert y_np.shape == (N, Dout)
    np.testing.assert_allclose(y_np, ref, rtol=1e-6, atol=1e-6)

#test backward
def test_linear_backward_1():
    """
    Backward test (parameter gradients):
      - Build a linear layer with bias, compute loss = mean(layer(x))
      - Run autograd backward to get dloss/dW and dloss/db
      - Compute numerical gradients for W and b via finite differences
      - Compare autograd gradients to numerical gradients within tolerances
    """
    N, Din, Dout = 4, 3, 5
    x0 = np.random.randn(N, Din).astype(np.float32)

    layer = Linear(Din, Dout, bias=True)
    W_param, b_param = _get_linear_params_by_shape(layer, Din, Dout, expect_bias=True)

    W0 = (np.random.randn(Din, Dout).astype(np.float32) * 0.1)
    b0 = (np.random.randn(Dout).astype(np.float32) * 0.1)
    W_param.data[...] = W0
    b_param.data[...] = b0

    x = Tensor(x0.copy(), requires_grad=True)
    y = layer(x)
    loss = y.mean()

    x.zero_grad()
    W_param.zero_grad()
    b_param.zero_grad()
    loss.backward()

    def fW(W_arr):
        W_param.data[...] = W_arr
        out = layer(Tensor(x0.copy(), requires_grad=False))
        return float(_np(out).mean())

    def fb(b_arr):
        b_param.data[...] = b_arr
        out = layer(Tensor(x0.copy(), requires_grad=False))
        return float(_np(out).mean())

    gW_num = _finite_diff_grad_param(fW, W_param.data.copy())
    gb_num = _finite_diff_grad_param(fb, b_param.data.copy())

    np.testing.assert_allclose(W_param.grad, gW_num, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(b_param.grad, gb_num, rtol=RTOL, atol=ATOL)
    
    
#test when bias==False
def test_linear_backward_2():
    """
    Backward test (no bias):
      - Build a linear layer without bias, compute loss = mean(layer(x))
      - Run autograd backward to get dloss/dW
      - Compute numerical gradient for W via finite differences
      - Compare autograd gradient to numerical gradient within tolerances
    """
    N, Din, Dout = 5, 4, 3
    x0 = np.random.randn(N, Din).astype(np.float32)

    layer = Linear(Din, Dout, bias=False)
    W_param, _ = _get_linear_params_by_shape(layer, Din, Dout, expect_bias=False)

    W0 = (np.random.randn(Din, Dout).astype(np.float32) * 0.1)
    W_param.data[...] = W0

    x = Tensor(x0.copy(), requires_grad=True)
    y = layer(x)
    loss = y.mean()

    x.zero_grad()
    W_param.zero_grad()
    loss.backward()

    def fW(W_arr):
        W_param.data[...] = W_arr
        out = layer(Tensor(x0.copy(), requires_grad=False))
        return float(_np(out).mean())

    gW_num = _finite_diff_grad_param(fW, W_param.data.copy())

    np.testing.assert_allclose(W_param.grad, gW_num, rtol=RTOL, atol=ATOL)
    
