import numpy as np
from .module import Module
from ..tensor import Tensor

"""
Example usage:
    >>> lin_layer = Linear(512, 256)  // Create an instance of the Linear layer with input feature size 512 and output feature size 256
    >>> x = ...  //A tensor of shape (batch_size, 512)
    //This will call the forward method of Linear, and compute the output tensor of shape (batch_size, 256) using tensor operations(like @, +, ...).
    >>> out = lin_layer(x)
"""
class Linear(Module):
    """
    A fully connected (affine) layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Construct a Linear layer.

        Inputs:
            in_features (int):
                Size of each input feature vector (last dimension of `x`).
            out_features (int):
                Size of each output feature vector (last dimension of `y`).
            bias (bool):
                If True, include a learnable bias term.

        Returns:
            None

        Side Effects:
            - Allocates and registers learnable parameters, including the weight and optionally the bias:
            - All learnable parameters must be created with `requires_grad=True`
              so they participate in autograd and appear in `parameters()`.
        """
        super().__init__()

        # TODO: define weight parameter as a member of this class instance (E.g., self.weights). It will be used in forward()
        # TODO: define bias parameter as a member of this class instance (E.g., self.bias) (if bias=True). It will be used in forward()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Linear layer.

        Inputs:
            x (Tensor):
                Input Tensor of shape (..., in_features).

        Returns:
            Tensor:
                Output Tensor of shape (..., out_features).

        Note:
            - Does not modify the input tensor, the weight and the bias in-place.
        """
        #TODO: - Compute the output of the linear layer using the input tensor, parameters(the weight and bias), and tensor operations.
        raise NotImplementedError
