import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pytest

from mini_torch.optim import SGD
from mini_torch.tensor import Tensor


def test_sgd_1():
    """
    Test SGD.step (basic parameter update):
      - With p.grad provided, step should perform: p <- p - lr * grad
      - Checks the updated parameter values match the expected NumPy result
    """
    p = Tensor(np.array([1.0, -2.0], dtype=np.float32), requires_grad=True)
    p.grad = np.array([3.0, -4.0], dtype=np.float32)  # dp

    opt = SGD([p], lr=0.1)
    opt.step()

    # p <- p - lr * grad
    expected = np.array([1.0, -2.0], dtype=np.float32) - 0.1 * np.array([3.0, -4.0], dtype=np.float32)
    np.testing.assert_allclose(p.data, expected, rtol=1e-6, atol=1e-6)



def test_sgd_2():
    """
    Test SGD.step when grad is None:
      - If a parameter has no gradient, SGD.step should skip updating it
      - Ensures parameter data remains unchanged
    """
    p = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p.grad = None

    opt = SGD([p], lr=0.1)
    before = p.data.copy()
    opt.step()

    np.testing.assert_allclose(p.data, before, rtol=0.0, atol=0.0)

def test_sgd_3():
    """
    Test SGD.zero_grad:
      - After calling zero_grad(), all parameter gradients should be cleared (set to None)
    """
    p1 = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    p2 = Tensor(np.array([2.0], dtype=np.float32), requires_grad=True)

    p1.grad = np.array([0.5], dtype=np.float32)
    p2.grad = np.array([-1.0], dtype=np.float32)

    opt = SGD([p1, p2], lr=0.1)
    opt.zero_grad()

    assert p1.grad is None
    assert p2.grad is None

