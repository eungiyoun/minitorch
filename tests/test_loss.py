import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pytest

import numpy as np

from mini_torch.tensor import Tensor
from mini_torch.nn.losses import MSELoss


def _allclose(a, b, atol=1e-6, rtol=1e-6):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

#forward test for MSELoss
def test_mse_loss_forward():
    pred0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target0 = np.array([[1.5, 1.0], [2.0, 5.0]], dtype=np.float32)

    pred = Tensor(pred0.copy(), requires_grad=True)

    criterion = MSELoss()
    loss = criterion(pred, target0)  

    expected = np.mean((pred0 - target0) ** 2)

    assert loss.shape == ()
    _allclose(loss.data, np.array(expected, dtype=np.float32))

#backward test for MSELoss
def test_mse_loss_backward_on_pred():
    pred0 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target0 = np.array([[1.5, 1.0], [2.0, 5.0]], dtype=np.float32)

    pred = Tensor(pred0.copy(), requires_grad=True)
    target = Tensor(target0.copy(), requires_grad=False)

    criterion = MSELoss()
    loss = criterion(pred, target)
    loss.backward()

    n = pred0.size
    expected_grad = (2.0 / n) * (pred0 - target0)

    assert pred.grad is not None
    _allclose(pred.grad, expected_grad.astype(np.float32))
    assert target.grad is None