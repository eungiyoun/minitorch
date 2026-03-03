import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
import pytest

from mini_torch.tensor import Tensor

def _allclose(a, b, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

'''
Test 1: gradient of a reduction (sum).
y = sum(x). For every element x_ij, dy/dx_ij = 1.
'''
def test_backward_1():

    x0 = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x0, requires_grad=True)

    y = x.sum()       
    y.backward()

    assert x.grad is not None
    np.testing.assert_allclose(x.grad, np.ones_like(x0, dtype=np.float32), atol=1e-6, rtol=1e-6)

'''
Test 2: pure chain rule across multiple sequential operations.
No shared nodes in the computation graph.

y = sum(((x * 2) + 3)^2)
For each element x_ij:

Let f(x) = ((2x + 3)^2)
Then df/dx = 2*(2x + 3) * 2 = 4*(2x + 3) = 8x + 12
'''
def test_backward_2():

    x0 = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)

    y = ((x * 2.0 + 3.0) * (x * 2.0 + 3.0)).sum()
    y.backward()

    ref = (8.0 * x0 + 12.0).astype(np.float32)

    assert x.grad is not None
    np.testing.assert_allclose(x.grad, ref, atol=1e-4, rtol=1e-4)


'''
Test 3: shared graph node (branch and merge).
x is used in two branches, then merged.

y = sum(2*x + 3*x) = sum(5*x)
So dy/dx = 5 for every element.
This tests whether gradients from multiple paths are accumulated correctly.
'''
def test_backward_3():

    x0 = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)

    y = (x * 2.0 + x * 3.0).sum()
    y.backward()

    ref = np.full_like(x0, 5.0, dtype=np.float32)

    assert x.grad is not None
    np.testing.assert_allclose(x.grad, ref, atol=1e-6, rtol=1e-6)
    
'''
Test 4: multiple backward calls should accumulate gradients
unless zero_grad() is called.

First backward:
    y1 = sum(x^2)
    dy/dx = 2x

Second backward:
    y2 = sum(3x)
    dy/dx = 3

After two backward calls without zero_grad:
    total grad = 2x + 3

After zero_grad:
    grad should reset.
'''
def test_backward_4():

    x0 = np.random.randn(2, 3).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)

    # First backward
    y1 = (x * x).sum()        # grad = 2x
    y1.backward()

    ref1 = (2.0 * x0).astype(np.float32)
    np.testing.assert_allclose(x.grad, ref1, atol=1e-4, rtol=1e-4)

    # Second backward (no zero_grad)
    y2 = (x * 3.0).sum()      # grad = 3
    y2.backward()

    ref2 = (2.0 * x0 + 3.0).astype(np.float32)
    np.testing.assert_allclose(x.grad, ref2, atol=1e-4, rtol=1e-4)

    # Reset gradients
    x.zero_grad()

    # Third backward (after reset)
    y3 = (x * 3.0).sum()      # grad = 3
    y3.backward()

    ref3 = np.full_like(x0, 3.0, dtype=np.float32)
    np.testing.assert_allclose(x.grad, ref3, atol=1e-4, rtol=1e-4)
    
    
'''
Test 5: node with requires_grad=False in the graph.

c does not require gradients.
y = sum(x * c)

Expected:
- x should receive gradient equal to c
- c.grad should remain None
'''
def test_backward_5():

    x0 = np.random.randn(2, 3).astype(np.float32)
    c0 = np.random.randn(2, 3).astype(np.float32)

    x = Tensor(x0.copy(), requires_grad=True)
    c = Tensor(c0.copy(), requires_grad=False)

    y = (x * c).sum()
    y.backward()

    # x receives gradient
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, c0, atol=1e-6, rtol=1e-6)

    # c should not receive gradient
    assert c.grad is None
    
    
# 6) test backward on freed graph raises or does not change grad   
def test_backward_6():
    rng = np.random.default_rng(8)
    x0 = rng.standard_normal((2, 2)).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)

    loss = (x * x).sum() 
    loss.backward()      
    g1 = x.grad.copy()

    try:
        loss.backward()    # should not propagate to x since graph was freed
    except Exception:
        # acceptable to raise; either way x.grad should remain unchanged
        pass

    _allclose(x.grad, g1)

    # If we re-forward, backward should accumulate again on leaf grads
    loss2 = (x * x).sum()
    loss2.backward()
    _allclose(x.grad, g1 + (2.0 * x0).astype(np.float32))