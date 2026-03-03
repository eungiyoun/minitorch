import numpy as np
from typing import Any, Optional, Tuple
from .tensor import Tensor, as_tensor


"""
Context object used to store information needed for backward computation.

It allows the forward() method to save intermediate tensors or values that
will be required later to compute gradients in backward().
"""
class Context:
    def __init__(self):
        self.saved_tensors: Tuple[np.ndarray, ...] = ()
        self.saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *xs: np.ndarray) -> None:
        self.saved_tensors = tuple(xs)

    def save_values(self, *vals: Any) -> None:
        self.saved_values = tuple(vals)


class Function:
    """
    Graph node: one instance per forward call.
    """
    def __init__(self, ctx: Context, parents: Tuple["Tensor", ...]):
        self.ctx = ctx
        self.parents = parents

    @staticmethod
    def forward(ctx: Context, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_out: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Any) -> "Tensor":
        """
        Apply this autograd Function to the given inputs.

        Inputs:
            *inputs:
                Positional inputs to the operation. Each input may be a Tensor,
                NumPy array, or Python scalar.

        Returns:
            Tensor:
                The output Tensor containing the forward result, with requires_grad
                set appropriately and grad_fn pointing to the creating Function
                when gradient tracking is enabled.
                
        Side Effects:
            - Sets parents and Context to connect the computation graph.
            - Executes the forward pass.
            - Attaches this Function instance to the output Tensor as grad_fn.
        """
        parents = tuple(as_tensor(x) for x in inputs)
        req = any(p.requires_grad for p in parents)
        # TODO:
        # 1) Create a Context Object, run cls.forward(ctx, ...) to compute the output value. 
            # The forward method in its subclass will compute the forward pass and store necessary information for backward in the Context Object.
            # Make sure to pass the raw data (np.ndarray) instead of Tensor to the forward method for numerical computation.
        
        # 2）Create the output Tensor (set data and requires_grad appropriately).
        
        # 3) Create the computation-graph node appropriately and attach it to the output Tensor (.grad_fn).
            # Note cls(the first argument) is the subclass of Function, so you can create the node by cls(...), with appropriate parameters.
        raise NotImplementedError


# ===== Functions (ops) =====
class Add(Function):
    """
    Forward:
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                First input array.
            b (np.ndarray):
                Second input array.

        Returns:
            np.ndarray:
                Element-wise sum of a and b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
    """
    @staticmethod
    def forward(ctx, a, b):
        # TODO: implement forward
        raise NotImplementedError
    """
    Backward:
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.
    """
    @staticmethod
    def backward(ctx, grad_out):
        # TODO: implement backward
        raise NotImplementedError

class Pow(Function):
    """
    Forward:
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                Base input array.
            b (np.ndarray):
                Exponent input array.

        Returns:
            np.ndarray:
                Element-wise power a ** b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
              required to compute gradients during the backward pass.
    """
    @staticmethod
    def forward(ctx, a, b):
        # TODO: implement forward
        raise NotImplementedError

    """
    Backward:
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.
    """
    @staticmethod
    def backward(ctx, grad_out):
        # TODO: implement backward
        raise NotImplementedError

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Inputs:
            ctx (Context):
                Context object for saving information needed in backward().
            a (np.ndarray):
                First input array.
            b (np.ndarray):
                Second input array.

        Returns:
            np.ndarray:
                Element-wise product of a and b. The output shape is the
                broadcasted shape of the inputs.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        # TODO: implement forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output, with the same shape as the forward output.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b. Each gradient has
                the same shape as its corresponding input.
        """
        # TODO: implement backward
        raise NotImplementedError


class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                Element-wise negation of x, with the same shape as x.
                
        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        # TODO: implement forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object from forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x, with the same shape as x.
        """
        # TODO: implement backward
        raise NotImplementedError


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        """
        Inputs:
            ctx (Context):
                Context object for saving backward information.
            a (np.ndarray):
                Left matrix operand.
            b (np.ndarray):
                Right matrix operand.

        Returns:
            np.ndarray:
                Matrix product of a and b.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        # TODO: implement forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Gradients with respect to inputs a and b.
        
        Note:
            You might find np.swapaxes useful here.
        """
        # TODO: implement backward
        raise NotImplementedError

#Example ops.
class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_values(x.shape)
        return np.array(x.sum(), dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        (sh,) = ctx.saved_values
        return (np.ones(sh, dtype=np.float32) * grad_out,)


class Mean(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                A scalar array containing the mean of all elements in x.

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        # TODO: implement forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the scalar output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x, with the same shape as x.
        """
        # TODO: implement backward
        raise NotImplementedError


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        """
        Inputs:
            ctx (Context):
                Context object for backward computation.
            x (np.ndarray):
                Input array.

        Returns:
            np.ndarray:
                Output array where each element is max(x, 0).

        Side Effects:
            - May store intermediate information in the Context object (ctx) that is
            required to compute gradients during the backward pass.
        """
        # TODO: implement forward
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x.
        """
        # TODO: implement backward
        raise NotImplementedError


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.empty_like(x, dtype=np.float32)

        pos_mask = x >= 0
        neg_mask = ~pos_mask

        out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

        exp_x = np.exp(x[neg_mask])
        out[neg_mask] = exp_x / (1.0 + exp_x)

        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Inputs:
            ctx (Context):
                Context object populated during forward().
            grad_out (np.ndarray):
                Gradient of the output.

        Returns:
            Tuple[np.ndarray]:
                Gradient with respect to input x.
        """
        # TODO: implement backward
        raise NotImplementedError
    
    
class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, logits, target):
        """
        logits: (N, C)
        target:
            - (N,)   class indices
            - (N, C) class probabilities (including one-hot)
        returns:
            scalar mean cross-entropy
        """
        if logits.ndim != 2:
            raise ValueError("CrossEntropy expects logits with shape (N, C).")

        N, C = logits.shape

        # stable log_softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        probs = exp_shifted / sum_exp
        log_probs = shifted - np.log(sum_exp)

        # Case 1: target is class indices, shape (N,)
        if target.ndim == 1:
            if target.shape[0] != N:
                raise ValueError("Target with class indices must have shape (N,).")

            target_idx = target.astype(np.int64)
            if np.any(target_idx < 0) or np.any(target_idx >= C):
                raise ValueError("Target contains invalid class index.")

            loss = -np.mean(log_probs[np.arange(N), target_idx]).astype(np.float32)
            target_dist = np.zeros((N, C), dtype=np.float32)
            target_dist[np.arange(N), target_idx] = 1.0

        # Case 2: target is class probabilities, shape (N, C)
        elif target.ndim == 2:
            if target.shape != (N, C):
                raise ValueError("Target probabilities must have shape (N, C).")

            target_dist = target.astype(np.float32)
            loss = -np.mean(np.sum(target_dist * log_probs, axis=1)).astype(np.float32)

        else:
            raise ValueError("Target must have shape (N,) or (N, C).")

        ctx.save_for_backward(probs.astype(np.float32), target_dist.astype(np.float32))
        ctx.save_values(N)
        return np.array(loss, dtype=np.float32)

    @staticmethod
    def backward(ctx, grad_out):
        probs, target_dist = ctx.saved_tensors
        (N,) = ctx.saved_values

        grad_logits = (probs - target_dist) / float(N)
        grad_logits = (grad_logits * grad_out).astype(np.float32)
        return grad_logits, None