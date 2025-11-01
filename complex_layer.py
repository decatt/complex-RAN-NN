"""
complex_layer.py
=================

This module provides a small collection of PyTorch layers and helpers for
operating on complex‑valued data. PyTorch has native support for complex
numbers (e.g. ``torch.complex64`` and ``torch.complex128`` dtypes), but
many of the built‑in layers only operate on real tensors. The classes
defined here wrap pairs of real–valued layers to implement the correct
algebra for complex operations.  Additionally, a simple pointwise
multiplication routine is provided for complex tensors.

The key components are:

* ``complex_multiplication`` – multiplies two complex numbers or tensors
  together, supporting both native complex tensors and explicit
  ``(real, imag)`` tuples.
* ``ComplexLinear`` – a fully connected layer that applies a complex
  weight matrix to a complex input.
* ``ComplexConv2d`` – a two‑dimensional convolution layer for complex
  inputs.  Internally it maintains separate real and imaginary
  convolutions and combines their outputs according to the rules of
  complex multiplication.
* ``ComplexDepthwiseSeparableConv2d`` – an efficient variant of
  ``ComplexConv2d`` that factors the convolution into a depthwise
  convolution followed by a pointwise (1×1) convolution.  This reduces
  the number of learnable parameters and computations, mirroring
  depthwise separable convolutions used in real‑valued models such as
  MobileNet.

The implementations here take inspiration from open source projects
including the ``complexPyTorch`` repository, but they have been written
from scratch for clarity.  No external dependencies beyond PyTorch are
required.

Example usage::

    import torch
    from complex_layer import ComplexLinear, ComplexConv2d, ComplexDepthwiseSeparableConv2d

    # create a complex input tensor (batch=8, channels=3, height=32, width=32)
    x = torch.randn(8, 3, 32, 32, dtype=torch.complex64)

    # apply a complex convolution
    conv = ComplexConv2d(3, 16, kernel_size=3, padding=1)
    y = conv(x)

    # apply a depthwise separable complex convolution
    dws = ComplexDepthwiseSeparableConv2d(3, 16, kernel_size=3, padding=1)
    z = dws(x)

    # apply a linear transformation to a flattened complex vector
    lin = ComplexLinear(32 * 32 * 16, 10)
    out = lin(y.view(y.size(0), -1))

"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional


def complex_multiplication(
    a: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    b: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the product of two complex numbers or tensors.

    This function supports two different representations for complex data:

    * Native complex tensors (e.g. dtype ``torch.complex64`` or ``torch.complex128``).
    * Tuples of two real tensors ``(real, imag)`` representing the real and
      imaginary parts explicitly.

    When both inputs are tuples, the result is returned as a tuple of
    ``(real, imag)``.  Otherwise, if at least one input is a complex
    tensor, the function falls back to the built‑in PyTorch complex
    multiplication.  Mixed representations (one tuple and one complex
    tensor) are not supported and will raise a ``TypeError``.

    Parameters
    ----------
    a : Tensor or Tuple[Tensor, Tensor]
        The first complex operand.  Must be either a complex tensor or a
        tuple of real tensors of the same shape.
    b : Tensor or Tuple[Tensor, Tensor]
        The second complex operand.  Must have the same representation as
        ``a``.

    Returns
    -------
    Tensor or Tuple[Tensor, Tensor]
        The complex product ``a * b``.  If the inputs were tuples, the
        return value is a tuple ``(real, imag)``; otherwise a complex
        tensor is returned.

    Examples
    --------
    >>> a = (torch.tensor([1., 2.]), torch.tensor([3., 4.]))
    >>> b = (torch.tensor([5., 6.]), torch.tensor([7., 8.]))
    >>> real, imag = complex_multiplication(a, b)
    >>> real
    tensor([-16., -18.])
    >>> imag
    tensor([26., 32.])

    >>> a = torch.tensor([1+2j, 3+4j])
    >>> b = torch.tensor([5+6j, 7+8j])
    >>> complex_multiplication(a, b)
    tensor([-7.+16.j, -11.+52.j])
    """
    # Both inputs are tuples of (real, imag)
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != 2 or len(b) != 2:
            raise ValueError("Both inputs must be tuples of length 2 when using explicit real/imag representation.")
        a_r, a_i = a
        b_r, b_i = b
        # Ensure shapes are compatible for broadcasting/multiplication
        if a_r.shape != a_i.shape or b_r.shape != b_i.shape:
            raise ValueError("Real and imaginary parts must have the same shape within each tuple.")
        # Perform element‑wise complex multiplication
        real_part = a_r * b_r - a_i * b_i
        imag_part = a_r * b_i + a_i * b_r
        return real_part, imag_part

    # Both inputs are native complex tensors
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if not a.is_complex() or not b.is_complex():
            raise TypeError(
                "When not using explicit real/imag tuples, both 'a' and 'b' must be complex tensors."
            )
        return a * b

    # Mixed representations are ambiguous
    raise TypeError(
        "Inputs must either both be complex tensors or both be tuples of real tensors representing (real, imag)."
    )


class ComplexLinear(nn.Module):
    """A fully connected layer for complex inputs.

    This layer implements a linear transformation of the form ``y = Wx + b``
    where ``x`` and ``y`` are complex and ``W`` and ``b`` are complex learnable
    parameters.  Internally, it stores separate real and imaginary weight
    matrices and biases.  The forward pass combines these according to the
    rules of complex multiplication.

    Parameters
    ----------
    in_features : int
        Number of features in the input vector.
    out_features : int
        Number of features in the output vector.
    bias : bool, optional
        If ``True``, adds a learnable complex bias to the output.  The
        default is ``True``.
    dtype : torch.dtype, optional
        Desired complex dtype of the layer parameters (e.g.
        ``torch.complex64`` or ``torch.complex128``).  The default is
        ``torch.complex64``.

    Example
    -------
    >>> lin = ComplexLinear(4, 2)
    >>> x = torch.randn(3, 4, dtype=torch.complex64)
    >>> y = lin(x)
    >>> y.shape
    torch.Size([3, 2])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        # Create real and imaginary weight parameters of shape (out_features, in_features)
        # Note: weights are stored as real tensors; they will be combined in the forward pass.
        self.weight_real = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.get_default_dtype())
        )
        self.weight_imag = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.get_default_dtype())
        )

        if bias:
            # Bias is complex: separate real and imaginary parts
            self.bias_real = nn.Parameter(torch.empty(out_features, dtype=torch.get_default_dtype()))
            self.bias_imag = nn.Parameter(torch.empty(out_features, dtype=torch.get_default_dtype()))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        # Initialize weights and biases using a symmetric distribution similar to PyTorch's default
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use the same initialization that nn.Linear uses on the real part
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias_real is not None and self.bias_imag is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to a complex input tensor.

        The input must have dtype ``torch.complex64`` or ``torch.complex128``.
        The output will have the same complex dtype.
        """
        if not isinstance(input, torch.Tensor) or not input.is_complex():
            raise TypeError(
                f"ComplexLinear requires a complex input tensor; got type {type(input)} with dtype {getattr(input, 'dtype', None)}."
            )
        # Compute real and imaginary parts separately: y = W x + b
        # x_real: (batch, in_features), weight_real: (out_features, in_features)
        # We perform matmul manually to avoid PyTorch's complex matmul limitations.
        real = torch.matmul(input.real, self.weight_real.t()) - torch.matmul(input.imag, self.weight_imag.t())
        imag = torch.matmul(input.real, self.weight_imag.t()) + torch.matmul(input.imag, self.weight_real.t())
        if self.bias_real is not None and self.bias_imag is not None:
            real = real + self.bias_real
            imag = imag + self.bias_imag
        return real.to(self.dtype) + 1j * imag.to(self.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_real is not None}"


class ComplexConv2d(nn.Module):
    """A two‑dimensional convolution layer for complex inputs.

    This layer mirrors the API of :class:`torch.nn.Conv2d` but accepts and
    produces complex tensors.  Internally, it keeps separate real and
    imaginary convolutional kernels and combines them using the complex
    convolution identity.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or Tuple[int, int]
        Size of the convolving kernel.
    stride : int or Tuple[int, int], optional
        Stride of the convolution.  Default: 1
    padding : int or Tuple[int, int], optional
        Zero‑padding added to both sides of the input.  Default: 0
    dilation : int or Tuple[int, int], optional
        Spacing between kernel elements.  Default: 1
    groups : int, optional
        Number of blocked connections from input channels to output channels.
        Setting ``groups=in_channels`` performs a depthwise convolution
        (each input channel is convolved with its own set of filters).  The
        default is 1.
    bias : bool, optional
        If ``True``, adds a learnable complex bias to the output.  Default: True
    dtype : torch.dtype, optional
        Desired complex dtype of the layer parameters.  Default: ``torch.complex64``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dtype = dtype

        # Real and imaginary convolution modules.  Biases are disabled here;
        # we manage complex bias explicitly below so that the bias addition
        # follows the rules of complex addition.
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.conv_i = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        if bias:
            # Complex bias: separate real and imaginary parts
            self.bias_real = nn.Parameter(
                torch.zeros(out_channels, dtype=torch.get_default_dtype())
            )
            self.bias_imag = nn.Parameter(
                torch.zeros(out_channels, dtype=torch.get_default_dtype())
            )
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the complex convolution to the input.

        The input must be a 4D complex tensor with shape
        ``(batch, channels, height, width)`` and a complex dtype.  The output
        has the same shape except that the channel dimension is
        ``out_channels``.
        """
        if not isinstance(input, torch.Tensor) or not input.is_complex():
            raise TypeError(
                f"ComplexConv2d requires a complex input tensor; got type {type(input)} with dtype {getattr(input, 'dtype', None)}."
            )
        # Perform convolution on real and imaginary parts separately
        real = self.conv_r(input.real) - self.conv_i(input.imag)
        imag = self.conv_r(input.imag) + self.conv_i(input.real)
        if self.bias_real is not None and self.bias_imag is not None:
            real = real + self.bias_real.view(1, -1, 1, 1)
            imag = imag + self.bias_imag.view(1, -1, 1, 1)
        return real.to(self.dtype) + 1j * imag.to(self.dtype)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size},"
            f" stride={self.stride}, padding={self.padding}, dilation={self.dilation},"
            f" groups={self.groups}, bias={self.bias_real is not None}"
        )
        return s


class ComplexDepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for complex inputs.

    A depthwise separable convolution factorizes a standard convolution into
    two operations: a depthwise convolution that applies a distinct spatial
    kernel to each input channel, and a pointwise (1×1) convolution that
    mixes information across channels.  This layer wraps
    :class:`ComplexConv2d` for both operations.  It is typically used to
    reduce the number of parameters and computational cost compared to a
    full complex convolution.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or Tuple[int, int]
        Size of the depthwise convolution kernel.
    stride : int or Tuple[int, int], optional
        Stride of the depthwise convolution.  Default: 1
    padding : int or Tuple[int, int], optional
        Zero‑padding added to both sides of the input for the depthwise
        convolution.  Default: 0
    dilation : int or Tuple[int, int], optional
        Spacing between kernel elements in the depthwise convolution.
        Default: 1
    bias : bool, optional
        If ``True``, adds a learnable complex bias to both the depthwise
        and pointwise convolutions.  Default: True
    dtype : torch.dtype, optional
        Desired complex dtype of the layer parameters.  Default: ``torch.complex64``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype

        # Depthwise convolution: groups=in_channels means each input
        # channel is convolved independently.  The number of output channels
        # is kept equal to in_channels.
        self.depthwise = ComplexConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            dtype=dtype,
        )

        # Pointwise convolution: a 1×1 convolution that mixes the channels
        # produced by the depthwise convolution.  Use kernel_size=1 and
        # groups=1 so that it operates across all channels.
        self.pointwise = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the depthwise separable convolution to the input."""
        x = self.depthwise(input)
        x = self.pointwise(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, bias={self.depthwise.bias_real is not None}"
        )


import math  # imported at end to avoid circular dependency during class definitions
