# MP1: Building `mini_torch`   A Minimal Autograd Library

## Introduction

In this MP, you will build a minimal deep-learning library named `mini_torch` using **NumPy only** (CPU-only). Your goal is to implement an end-to-end, PyTorch-style training workflow: define `Tensor` objects, construct a dynamic computation graph during the forward pass, perform backpropagation, and train small neural networks.

By the end of this MP, your library should be able to:
- Correctly implement both the forward pass and the backward pass for each operation.
- Build a dynamic computation graph during the forward pass and use it to propagate gradients in reverse order during `backward()`.
- Provide a small neural network interface (e.g., a `Linear` layer, activation functions, and a loss such as `MSELoss`) for composing models.
- Train models by updating learnable parameters with an optimizer (e.g., SGD) using the standard training pattern: `zero_grad()`, `loss.backward()`, and `step()`.

**Important:** For detailed task descriptions and example usage, refer to the inline comments in the code. We also recommend that you read through the entire starter codebase before starting your implementation.

 
## Environment Setup and Library Policy

This MP (`mini_torch`) is a **CPU-only** minimal deep-learning library implemented using **NumPy only**. Please use **Python 3.10+** (recommended) and create an isolated environment to avoid dependency conflicts.

### Install required packages

Install NumPy and pytest with the specified versions:
- `pip install numpy==2.2.0`
- `pip install pytest==9.0.2`

### Allowed / Disallowed Libraries

**Allowed:** Use **NumPy** for tensor computation, plus **Python standard library** for utilities (e.g., `math`, `random`, `typing`, `dataclasses`, `itertools`, `functools`, `collections`, `os/pathlib`, `time/timeit`).

**Disallowed:** Do **NOT** use PyTorch or any other third-party deep learning or automatic differentiation libraries (e.g., `torch*`, `jax*`, `tensorflow/keras`, `autograd`, `cupy`). The only exception is that you may *temporarily* import and use `torch` in `mnist_classification.py` for comparison or testing purposes; however, all `torch` usage and imports must be removed from `mnist_classification.py` before the final submission.

 
## Explanation of Autograd and Computation Graphs (in *mini_torch*)

### What is Autograd?
Autograd (automatic differentiation) enables `mini_torch` to compute gradients automatically from a scalar output by applying the chain rule over a dynamically recorded computation graph. Computation graph nodes record how outputs depend on inputs, and `backward()` propagates gradient information from the output back to all reachable `Tensor`s with `requires_grad=True`. 

 
### What is a Computation Graph?
A computation graph is a directed acyclic graph (DAG) that represents a mathematical expression, where nodes correspond to operations or variables, and  edges represent the flow of data between them.

In this MP:
- **Nodes:** `Function` instances created by `Function.apply(...)`. Each differentiable operation in the computation graph creates one such node. Concretely, operations such as `Add`, `Mul`, `MatMul`, `Sum`, and `Mean` are subclasses of `Function`. When an operation is applied, its corresponding `Function` instance is created and attached to the output tensor’s `grad_fn`.

- **Edges:** references stored in `Function.parents` that point to the input `Tensor`s of that operation.

During the forward pass, the graph is constructed dynamically as operations execute. During the backward pass,
starting from a scalar output, gradients are propagated backward through the recorded connections to compute
and accumulate gradients for all required tensors.


### A Concrete Example
```python
a = x * w
c = a + x
y = c.sum_all()


Computation graph view (Function = node):

(Tensor x) ─┐
            ├─>[ Mul ](Tensor a)────┐
(Tensor w) ─┘                       ├─> [ Add ](Tensor c)─>[ Sum ](Tensor y)
(Tensor x) ─────────────────────────┘

Backward flow:

start:  dy/dy = 1  (initialize y.grad = 1)

(Tensor y.grad=1)
      |
      v
[ Sum.backward ]  ──> produces grad for (Tensor c)
      |
      v
[ Add.backward ]  ──> splits grad to (Tensor a) and (Tensor x)   (two paths)
      |                         |
      v                         v
[ Mul.backward ]      adds contribution to x.grad
      |
      ├──> adds contribution to x.grad   (from the mul path)
      └──> adds contribution to w.grad
```

**The following section describes the specific tasks.**
## Task #1 (`mini_torch/ops.py`)

### Compution graph node

In this MP, the computation graph is built from `Function` operation nodes. You need to complete the `forward(...)` and `backward(...)` implementations for the following nodes:

- `Add` (elementwise addition)
- `Mul` (elementwise multiplication)
- `Neg` (unary negation)
- `MatMul` (matrix multiplication via `@`, following NumPy broadcasting rules)
- `Pow` (power)
- `Mean` (reduction mean, to a scalar)
- `ReLU` (activation)
- `Sigmoid` (activation)

In this MP, the computation graph is constructed from **operation nodes** (`Function` instances). To store any information needed for the backward pass, each node contains a per-operation **Context** object (`ctx`), which is created during the forward pass.

Below is a high-level explanation of why information is stored for the backward pass, similar to the corresponding part in the PyTorch Tutorial lecture. For concrete code examples, you may refer to the `Sum` and `Context` implementations.

Example: 
```python
forward: x (Tensor, shape = ...)
                │
                │  forward: y = x.sum()
                ▼
          [   Sum   ]  ◀──────────────────────────────────────────
                │                                                │
                ▼                                                │
        y (Tensor, scalar)                                       │
                                                      (How do we know the shape of x?
                                                       We should store it in Context during forward!)
backward: dL/dy  (scalar)                                        │
                │                                                │
                │  backward:                                     │
                │  dL/dx = 1 * dL/dy                             │
                │  (broadcast to x.shape) ────────────────────────
                ▼
          [   Sum   ]
                │
                ▼
        dL/dx  (same shape as x)

```
For different operations, you may want to store the inputs (or other information) for different purposes.

For example:

- For **ReLU**, you may need to know which elements are greater than 0 and which are less than or equal to 0, so that the correct gradient can be computed during the backward pass (this could be stored as a mask or as the full input).

- For **Mul**, you may need to know the actual **input values**, since the gradient depends directly on them (e.g., ∂(a·b)/∂a = b and ∂(a·b)/∂b = a).

### Remark
  1. `__add__`, `__mul__`, and similar methods in `tensor.py` define the behavior of Python operators. For example, `a + b` calls `a.__add__(b)`, and `a * b` calls `a.__mul__(b)`. By overriding these methods, basic arithmetic operators can be used not only to perform numerical computation, but also to construct the computation graph.
  2. The implementation of the `Context` class and one example (`Sum`) has already been provided to you as a reference. 

### Specific task

- **Complete `Function.apply(...)`.**  
  `Function.apply(...)` is the primary entry point for graph construction. `Tensor` overloads operators such as `+`, `-`, `*`, and `@`, and these operators should dispatch to the corresponding `Function.apply(...)`. `apply(...)` should:
  1. Perform the computation and caching: create a `Context` object, run the operation’s `forward(...)` method to compute the output value, and store any information needed for the backward pass in the `Context` object.
  2. Construct the output `Tensor`.
  3. Construct the computation graph: create the computation-graph node with its `ctx` and `parents` (e.g., by calling `cls(...)`), and attach the operation node to the output `Tensor` (e.g., via `out.grad_fn`) **when gradient tracking is needed**.

- **Implement `forward` and `backward` for each operation.**  
  For every operation node (you need to complete `Add`, `Mul`, `Neg`, `MatMul`, `Pow`, and `Mean` at this moment.):
  - `forward(ctx, ...)` computes the numerical output and saves any necessary information into `ctx`;
  - `backward(ctx, grad_out)` computes gradients with respect to each input, returning them in the same order as the inputs, so gradients can be propagated and accumulated correctly;
  - For `Mean`, To reduce difficulty, you do not need to support parameters such as `axis`; only implement the default behavior, i.e., reducing over **all** elements of the input tensor.
  - **Type requirement:** all intermediate values and returned gradients should be stored/returned as `float32` (e.g., use `np.float32` / `astype(np.float32)` where appropriate).
- **Important:** For numerical computation, tensor operations and NumPy operations should be clearly distinguished. In other words, in parts of the code that perform numerical computations (such as `+`, `-`, `*`, etc.), you should use `Tensor.data` to access the underlying NumPy array rather than operating directly on the tensor itself when appropriate. You should always be mindful of which data type should be used, and some type hints are provided in the code. This is necessary to avoid abnormal computational graph behavior.

### Some Cases to Consider

To receive full credit, make sure you have considered all three cases.

1. **Broadcasting-aware gradients.** All of your operators should support NumPy-style broadcasting. In the backward pass, if an input tensor was broadcast in the forward pass, its gradient must sum contributions over all broadcasted positions.

   Example:  
   Let `x` be shape `(3,)` and `y` be a scalar:

   ```python
   x = np.array([1., 2., 3.])  # shape (3,)
   y = 10.0                    # scalar
   z = x + y                   # y is broadcast to shape (3,)
   ```

   Suppose the upstream gradient is `g = dL/dz` with shape `(3,)`.
   Then ∂L/∂y = Σᵢ gᵢ, i.e., you must sum along the broadcasted dimension.
2. **Constant inputs.**  
   If a constant input appears in the computation graph, you should convert it into a `Tensor` with `requires_grad=False`. In this case, the constant input must not receive gradients.

3. **Ensure that all operations support tensors of arbitrary dimensionality.**  
   The following is a high-level guideline for implementing `__matmul__`, covering vector, matrix, and batched matrix multiplication in a unified way.

   #### Forward (Shape Rules)

   Follow NumPy’s `matmul` broadcasting rules:

   ```text
   (..., m, n) @ (..., n, k) -> (..., m, k)
   ```

   This rule uniformly covers:

   - Vector · Vector: `(n,) @ (n,) -> ()`
   - Matrix · Vector: `(m, n) @ (n,) -> (m,)`
   - Vector · Matrix: `(n,) @ (n, k) -> (k,)`
   - Matrix · Matrix: `(m, n) @ (n, k) -> (m, k)`
   - Batched matmul: `(..., m, n) @ (..., n, k) -> (..., m, k)`

   #### Backward (Gradient Rules)

   Let:

   ```text
   u = a @ b
   dL/du = grad_out      with shape (..., m, k)
   ```

   Then:

   ```text
   dL/da = grad_out @ np.swapaxes(b, -1, -2)
   dL/db = np.swapaxes(a, -1, -2) @ grad_out
   ```

   The transpose applies only to the last two dimensions:

   ```text
   (..., m, n) -> (..., n, m)
   ```

   Internally, you may temporarily reshape 1D inputs to 2D (e.g., `(1, n)` or `(n, 1)`) to unify the implementation, and reshape the result back afterward.

   These formulas apply uniformly to 2D matmul, batched matmul, and vector cases (after temporarily promoting to 2D).

   If broadcasting occurred in the leading (batch) dimensions during the forward pass, you must additionally reduce (sum) gradients over the broadcasted axes so that the gradients match the original shapes of `a` and `b`. 

   **Important:** For other operations (including `Add`, `Mul`, `Neg`, `ReLU`, `Sigmoid`, and `Pow`), you must also ensure that they support tensors of arbitrary dimensionality.

   In general, these operations are element-wise. If your implementation does not explicitly assume 2D inputs (e.g., by hard-coding shape unpacking, writing dimension-specific loops, or reshaping tensors to fixed ranks), then they should naturally generalize to tensors of arbitrary dimensionality without requiring additional special handling.

After implementing each operator, you can run the operator-specific tests using:
```bash
pytest -q --op add
```
Replace `add` with other operator names such as `neg`, `mul`, `matmul`, or `pow` to test the corresponding implementation.

## Task #2 (`mini_torch/tensor.py`)
### Do backpropagation over the computation graph.

Note each `Function` node is attached to the output `Tensor` via `Tensor.grad_fn`, and its `parents` attribute links to the input (parent) Tensors.
Starting from a scalar output (e.g., a loss), call `backward()` to propagate gradients backward through the graph by following
the links recorded during the forward pass, and accumulate gradients into all reachable leaf tensors. Your task is to complete `backward()` function of `Tensor` class.

### A suggested workflow:
1. Starting from the output tensor, collect all tensors involved in the computation by following `grad_fn` and each node’s `parents`, so that you can identify the full computation graph needed for backpropagation.
2. Traverse these tensors in an order that respects dependencies. For each tensor, use the gradient it receives to compute gradients for its parents, then propagate and accumulate those gradients appropriately.
3. After backpropagation is complete, free the computation graph by default so intermediate references do not remain unnecessarily.

### Note:
- If a tensor receives no upstream gradient in the current backward pass (i.e., `grad_out is None`), skip it and do not propagate gradients.
- Leaf tensors with `requires_grad=False` must never receive gradients. Their `.grad` attribute should remain `None` before and after `backward()`, and must not be modified.
- In `mini_torch`, gradients for **leaf tensors** with `requires_grad=True` are stored in `.grad`.  
  These gradients **accumulate across multiple `backward()` calls** unless they are explicitly cleared (e.g., via `zero_grad()`).

  "Accumulate" means that each new gradient is **added to the existing value in `.grad`**, rather than overwriting it.

  Example:

  ```python
  x = Tensor([2.0], requires_grad=True)

  y1 = x * x          # y1 = x^2
  y1.backward()       # dy1/dx = 2x = 4
  print(x.grad)       # 4

  y2 = x * 3          # y2 = 3x
  y2.backward()       # dy2/dx = 3
  print(x.grad)       # 7  (4 + 3)
- **Intermediate (non-leaf) tensors do not store gradients in `.grad` by default.** Their gradients are computed only as temporary values during a single backward pass and are not retained after backpropagation (should be `None`). This design avoids unnecessary memory usage and helps prevent unintended behavior.
- After backpropagation, the **computation graph should be released by default** by breaking the references that keep it alive (e.g., clearing each non-leaf tensor’s `grad_fn`, clearing each `Function` node’s `parents`, and clearing any saved data stored in `ctx`). Backpropagating through a freed computation graph should not affect the gradients of the leaf nodes in that graph.
- If a node has multiple child nodes, make sure it receives gradients from all of them and accumulates them.
- If a node has requires_grad = False, it should not receive gradients (its .grad should remain None).
To test your implementation, run:
```bash
pytest -q tests/test_backward.py
```

So far, you have implemented the core autograd engine for `mini_torch`. Next, you will build additional components needed
to construct and train neural networks.


## Task #3 (`mini_torch/nn/layers.py`)

### Module
A `Module` is a reusable neural-network component that wraps a forward computation and manages its trainable parameters.
In `mini_torch`, `MSELoss`, the `Linear` Layer, and activation functions are implemented as `Module`s. The code for `Module` has already been provided.


### Linear layer
A linear layer applies an affine transformation:

$$
y = xW + b.
$$

### Example Usage
```python
lin = Linear(in_features=4, out_features=3, bias=True)
```
### Implementing the Linear layer
Implement the `Linear` layer by completing:

- **`__init__(in_features, out_features, bias=True)`**  
  Initialize the layer’s trainable parameters using `Tensor`:
  - Create a trainable weight parameter with shape `(in_features, out_features)` and set `requires_grad=True`.
  - If `bias=True`, create a trainable bias parameter with shape `(out_features,)` and set `requires_grad=True`.

**Important:** The layer must contain exactly one `weight` Tensor and, if enabled, exactly one `bias` Tensor, both strictly following the required shapes. Do not add any extra trainable parameters.

- **`forward(x)`**  
  Compute the forward pass using previously implemented `Tensor` operators:
  - Multiply the input by the layer’s weight parameter to produce the linear output.
  - If a bias parameter exists, add it to the output using broadcasting.

**Important:** Ensure that you use the overloaded `Tensor` operators (e.g., `+`, `-`, etc.) instead of directly modifying the `Tensor.data` field, as doing so will break the computation graph.

To test your Linaer layer implementation, run the unit tests from the project root:
```bash
pytest -q tests/test_linear.py
```

## Task #4 (`mini_torch/ops.py`)

### Activation functions

Activation functions apply a non-linear transformation to a tensor. They are essential in neural networks because they introduce non-linearity; 
In `mini_torch`, activation functions are implemented as `Module`s and are used by calling their `forward` method on an input tensor.

In each activation module’s `forward()` method (see `mini_torch/nn/activations.py`), you should call the corresponding operation node (e.g., `ReLU` and `Sigmoid`) defined in `ops.py`.
To support autograd, you must implement the `forward(ctx, x)` and `backward(ctx, grad_out)` methods for these nodes, consistent with how you implemented other operators.
For `Sigmoid`, the forward implementation has already been given to you. You only need to finish the backward part.
By now, you should already be quite familiar with implementing operators...

Do not forget to verify the fundamental functionality of each operator by running the operator-specific tests, e.g.:
```bash
pytest -q --op relu(or sigmoid)
```

 



## Task #5 (`mini_torch/nn/losses.py`)

### MSELoss
Loss functions such as `MSE loss` and `cross-entropy loss` are commonly used in neural network training. The implementation of the `CrossEntropyLoss` is provided to you as a reference.
A loss is a **scalar** that measures how well the model predictions match the targets. During training, you minimize the loss:
backpropagation computes gradients of the loss with respect to model parameters, and the optimizer updates parameters to reduce the loss.

Mean Squared Error (MSE) is defined as

$$
\mathrm{MSE}(\texttt{pred}, \texttt{target}) = \frac{1}{N}\sum_{i=1}^{N}(\texttt{pred}_i - \texttt{target}_i)^2,
$$

where $N$ is the number of elements.

### Example Usage
```python
loss_fn = MSELoss()
loss = loss_fn(pred, target)  
loss.backward()              
```

### Implementing the MSELoss
Implement `MSELoss.forward(pred, target)` to return a **scalar `Tensor`** MSE loss. If `target` is not a `Tensor`, convert it to a constant `Tensor` (`requires_grad=False`) so autograd tracks gradients only through `pred`.


## Task #6 (`mini_torch/optim/sgd.py`)

### Optimizer
An optimizer updates a model's trainable parameters using their gradients. After backpropagation computes `param.grad`, the optimizer applies an update rule (e.g., SGD) to modify `param.data` and reduce the loss over training steps.

### Example usage
```python
...
opt = SGD(model.parameters(), lr=1e-2)

...
opt.zero_grad()
loss.backward()
opt.step()
```

### Implement the optimizer methods `step()` and `zero_grad()` in `SGD`.

- **`step()`:** Perform one optimization step. Iterate over `self.params`. For each parameter `p`,
  skip it if `p.grad` is `None`; otherwise update the parameter *in place* using SGD:

  ```text
  p = p - lr * grad
  ```

- **`zero_grad()`:** Clear gradients before the next backward pass. Since gradients in this MP are *accumulated*,
  you must reset each parameter's `grad`, so that the next call to `backward()`
  starts from a clean state.

To test your SGD implementation, run the unit tests from the project root:
```bash
pytest -q tests/test_sgd.py
```

## Task #7(`mnist_classification.py`)
MNIST Classification (MLP)

### Dataset
- MNIST is provided locally in `data/mnist/` as IDX gzip files (`*.gz`); no download is needed.
- The starter code already implements loading, preprocessing, and construction of `TensorDataset`/`DataLoader`.

### Model (Implement `MNISTMLP`)
Implement a neural network for MNIST classification using `mini_torch.nn`. You may freely design the architecture (e.g., number of layers, hidden dimensions, and activation functions). For the loss function section, although MSELoss is not the best choice for classification, we use it in this MP to reduce the difficulty.

Implement the model in the `MNISTMLP` class by completing:
- `__init__(...)`: define and store the layers/modules
- `forward(x)`: implement the forward computation and return the model output
- The input dimenstion is 784 and the output dimension is 10(10 classes).

### Training Loop
1. **Initialize** the model, loss function (criterion, already provided), and optimizer.
2. **For each minibatch**:
   - Run the **forward pass** to obtain predictions.
   - Compute the **loss** from predictions and targets.
   - **Reset accumulated gradients and backpropagate** the current loss to obtain gradients for all learnable parameters.
   - **Update parameters**.

**To receive credit, the final test accuracy must exceed 0.95** (i.e., > 0.95).

Note that, in order for the autograder to grade your submission successfully, you should avoid excessively long training times.

### Cross-framework Validation

To test your implementation, you can temporarily switch the imports in `mnist_classification.py` from `mini_torch` to standard PyTorch and compare whether training produces similar loss/accuracy trends.

Change:

```python
import mini_torch as torch
import mini_torch.nn as nn
from mini_torch.utils.data import TensorDataset, DataLoader
from mini_torch.optim import SGD
from mini_torch import Tensor
from mini_torch.nn import CrossEntropyLoss

```
to

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch import Tensor
from torch.nn import CrossEntropyLoss
```

**Notes:**
1. Due to differences such as parameter initialization, the loss and accuracy at each step do not need to match exactly. However, the overall training trend and the stabilized (final) loss should be generally similar.
2. **IMPORTANT: Before submitting your assignment, make sure to switch the imports back to the `mini_torch` version.**

## Note about Tests

Before submitting, you can run all public tests with:

```bash
pytest -q tests
```

Passing all public tests—even if training appears to work—does **not** guarantee full credit. Your implementation will also be evaluated with **additional hidden tests**, but **only for the functionality explicitly described in the documentation**.

## Submission Instructions

Please submit **one single `.zip` file** containing **exactly** the following:

1. The **`mini_torch/`** folder (your full implementation)
2. **`mnist_classification.py`** (the MNIST training script)

Please **exclude** the following files and folders from your submission:

- `__pycache__/`
- `*.pyc`
- `*.pyo`
- `.pytest_cache/`
- `.ipynb_checkpoints/` (if any)

### Create the zip file (**include only the required files and exclude all unnecessary files**)

**On Linux/macOS**, you can run:

```bash
find mini_torch mnist_classification.py -type f -name "*.py" | zip -@ submission.zip
```

**On Windows**, you can run:

```bat
cmd /c "rmdir /s /q _s 2>nul & robocopy mini_torch _s\mini_torch *.py /S & copy mnist_classification.py _s\ & tar -a -c -f submission.zip -C _s mini_torch mnist_classification.py"
```

You may also delete unnecessary files manually, then create a `.zip` file containing only the required folder and file.

### After Submission

Please check the feedback on Gradescope after each submission.

If your submission is successful, you will receive a score immediately for the **public portion** of the tests (up to **60/100**). This is **not** the final score for the MP.

If you see:

**"Submission rejected: failed file-format pre-check."**

please revise your submission format according to the specific error message shown on Gradescope, and then submit again.

> **Important:** Always review the Gradescope feedback after every submission to make sure your submission format is correct.

## Contact Information for MP Questions

If you have any questions about the MP, please feel free to reach out to:

**Shanbin Sun**  
**Email:** `shanbin3@illinois.edu`
