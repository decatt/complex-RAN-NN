from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Union, Optional
import torch.nn.functional as F

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

def complex_gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation separately to real and imaginary parts.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor to activate.

    Returns
    -------
    torch.Tensor
        Complex tensor with GELU applied to real and imaginary parts.
    """
    return F.gelu(x.real) + 1j * F.gelu(x.imag)


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex inputs.

    This module wraps two real ``nn.LayerNorm`` instances to
    independently normalize the real and imaginary components of a
    complex tensor.  It expects inputs of shape ``(..., C)`` where
    ``C`` is the number of channels.  The normalization is applied
    across the last dimension as in ``nn.LayerNorm``.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.norm_imag = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexLayerNorm expects a complex tensor, got dtype {x.dtype}"
            )
        real = self.norm_real(x.real)
        imag = self.norm_imag(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexPixelShuffle(nn.Module):
    """PixelShuffle for complex tensors.

    Applies ``nn.PixelShuffle`` to the real and imaginary parts
    separately and recombines them into a complex tensor.  This is
    commonly used in upsampling operations.
    """

    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self._real_shuffle = nn.PixelShuffle(upscale_factor)
        self._imag_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexPixelShuffle expects a complex tensor, got dtype {x.dtype}"
            )
        real = self._real_shuffle(x.real)
        imag = self._imag_shuffle(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexPixelUnshuffle(nn.Module):
    """PixelUnshuffle for complex tensors.

    Applies ``nn.PixelUnshuffle`` to the real and imaginary parts
    separately and recombines them into a complex tensor.  This is
    commonly used in downsampling operations.
    """

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor
        self._real_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self._imag_unshuffle = nn.PixelUnshuffle(downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexPixelUnshuffle expects a complex tensor, got dtype {x.dtype}"
            )
        real = self._real_unshuffle(x.real)
        imag = self._imag_unshuffle(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexDownSample(nn.Module):
    """Downsample a complex tensor by a factor of 2.

    The operation consists of a complex convolution that halves the
    number of channels followed by a pixel unshuffle that reduces the
    spatial dimensions by a factor of 2 and increases channels by a
    factor of 4.  This mirrors the real‐valued downsampling module in
    the original Restormer.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # First reduce channels by half using a complex 3×3 convolution
        self.conv = ComplexConv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False)
        # Then apply pixel unshuffle to downsample height and width by 2 and
        # increase channels by a factor of 4
        self.unshuffle = ComplexPixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unshuffle(self.conv(x))


class ComplexUpSample(nn.Module):
    """Upsample a complex tensor by a factor of 2.

    The operation consists of a complex convolution that doubles the
    number of channels followed by a pixel shuffle that increases the
    spatial dimensions by a factor of 2 and reduces channels by a
    factor of 4.  This mirrors the real‐valued upsampling module in
    the original Restormer.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # First expand channels by a factor of 2 using a complex 3×3 convolution
        self.conv = ComplexConv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        # Then apply pixel shuffle to upsample height and width by 2 and
        # reduce channels by a factor of 4
        self.shuffle = ComplexPixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class ComplexMDTA(nn.Module):
    """Complex multi‐DConv head transposed self‐attention (MDTA).

    This is a complex variant of the MDTA block described in the
    Restormer implementation.  Queries, keys and
    values are generated using complex convolutions; attention weights
    are computed from the real parts of the complex dot products
    between normalized queries and conjugate keys.  The resulting
    weights (real) are applied to the values to produce the output.
    """

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("'channels' must be divisible by 'num_heads'.")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        # Generate queries, keys and values
        self.qkv = ComplexConv2d(channels, channels * 3, kernel_size=1, bias=False)
        # Depthwise convolution to mix local context
        self.qkv_conv = ComplexConv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        # Project the output back to the original channel dimension
        self.project_out = ComplexConv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexMDTA expects a complex input tensor, got dtype {x.dtype}"
            )
        b, c, h, w = x.shape
        # Generate q, k, v
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        # Reshape to (batch, heads, channels_per_head, spatial_positions)
        c_head = c // self.num_heads
        # q, k, v have shape (b, c, h*w) when reshaped; but we need (b, heads, c_head, h*w)
        q = q.view(b, self.num_heads, c_head, h * w)
        k = k.view(b, self.num_heads, c_head, h * w)
        v = v.view(b, self.num_heads, c_head, h * w)
        # Normalize q and k along the last dimension (tokens) as in the original MDTA
        # Compute the magnitude squared across real and imaginary parts
        q_mag = (q.real**2 + q.imag**2) + 1e-6
        k_mag = (k.real**2 + k.imag**2) + 1e-6
        # Normalize across the token dimension
        q_norm = torch.linalg.norm(q_mag, dim=-1, keepdim=True)
        k_norm = torch.linalg.norm(k_mag, dim=-1, keepdim=True)
        q = q / (q_norm + 1e-6)
        k = k / (k_norm + 1e-6)
        # Compute complex dot product of q and conjugate of k along the token dimension
        # The real part of the dot product is used to build the attention scores
        # conj(k) = k.real - i k.imag; real part of q * conj(k) is (q.real*k.real + q.imag*k.imag)
        # We perform matrix multiplication along the token dimension (-1)
        real_scores = torch.matmul(q.real, k.real.transpose(-2, -1)) + torch.matmul(q.imag, k.imag.transpose(-2, -1))
        # Apply temperature and softmax to obtain attention weights
        attn = torch.softmax(real_scores * self.temperature, dim=-1)
        # Weighted sum of values
        # v has shape (b, heads, c_head, h*w).  attn: (b, heads, c_head, c_head)
        out_real = torch.matmul(attn, v.real)
        out_imag = torch.matmul(attn, v.imag)
        out = out_real + 1j * out_imag
        # Reshape back to (b, c, h, w)
        out = out.view(b, c, h, w)
        # Final projection
        out = self.project_out(out)
        return out


class ComplexGDFN(nn.Module):
    """Complex gated depthwise feed–forward network (GDFN).

    This is a complex variant of the GDFN block described in the
    Restormer implementation.  It uses complex
    convolutions to project the input to a higher dimensional space,
    applies a depthwise convolution and gating via a complex GELU, and
    then projects back to the original channel dimension.
    """

    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        # Project input to 2 × hidden channels
        self.project_in = ComplexConv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        # Depthwise convolution on the projected tensor
        self.conv = ComplexConv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        # Project back to original channel dimension
        self.project_out = ComplexConv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexGDFN expects a complex input tensor, got dtype {x.dtype}"
            )
        # Project and split into two halves
        x_proj = self.conv(self.project_in(x))
        x1, x2 = torch.chunk(x_proj, chunks=2, dim=1)
        # Apply complex GELU to the first half
        x1 = complex_gelu(x1)
        # Element‐wise complex multiplication with the second half
        x_combined = x1 * x2
        # Project back to the original channel dimension
        return self.project_out(x_combined)


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

def complex_gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation separately to real and imaginary parts.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor to activate.

    Returns
    -------
    torch.Tensor
        Complex tensor with GELU applied to real and imaginary parts.
    """
    return F.gelu(x.real) + 1j * F.gelu(x.imag)


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex inputs.

    This module wraps two real ``nn.LayerNorm`` instances to
    independently normalize the real and imaginary components of a
    complex tensor.  It expects inputs of shape ``(..., C)`` where
    ``C`` is the number of channels.  The normalization is applied
    across the last dimension as in ``nn.LayerNorm``.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.norm_imag = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexLayerNorm expects a complex tensor, got dtype {x.dtype}"
            )
        real = self.norm_real(x.real)
        imag = self.norm_imag(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexPixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self._real_shuffle = nn.PixelShuffle(upscale_factor)
        self._imag_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexPixelShuffle expects a complex tensor, got dtype {x.dtype}"
            )
        real = self._real_shuffle(x.real)
        imag = self._imag_shuffle(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexPixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor
        self._real_unshuffle = nn.PixelUnshuffle(downscale_factor)
        self._imag_unshuffle = nn.PixelUnshuffle(downscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexPixelUnshuffle expects a complex tensor, got dtype {x.dtype}"
            )
        real = self._real_unshuffle(x.real)
        imag = self._imag_unshuffle(x.imag)
        return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexDownSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # First reduce channels by half using a complex 3×3 convolution
        self.conv = ComplexConv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False)
        # Then apply pixel unshuffle to downsample height and width by 2 and
        # increase channels by a factor of 4
        self.unshuffle = ComplexPixelUnshuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unshuffle(self.conv(x))


class ComplexUpSample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # First expand channels by a factor of 2 using a complex 3×3 convolution
        self.conv = ComplexConv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False)
        # Then apply pixel shuffle to upsample height and width by 2 and
        # reduce channels by a factor of 4
        self.shuffle = ComplexPixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class ComplexMDTA(nn.Module):
    """Complex multi‐DConv head transposed self‐attention (MDTA).

    This is a complex variant of the MDTA block described in the
    Restormer implementation 【802216581475939†L4-L29】.  Queries, keys and
    values are generated using complex convolutions; attention weights
    are computed from the real parts of the complex dot products
    between normalized queries and conjugate keys.  The resulting
    weights (real) are applied to the values to produce the output.
    """

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("'channels' must be divisible by 'num_heads'.")
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        # Generate queries, keys and values
        self.qkv = ComplexConv2d(channels, channels * 3, kernel_size=1, bias=False)
        # Depthwise convolution to mix local context
        self.qkv_conv = ComplexConv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        # Project the output back to the original channel dimension
        self.project_out = ComplexConv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexMDTA expects a complex input tensor, got dtype {x.dtype}"
            )
        b, c, h, w = x.shape
        # Generate q, k, v
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        # Reshape to (batch, heads, channels_per_head, spatial_positions)
        c_head = c // self.num_heads
        # q, k, v have shape (b, c, h*w) when reshaped; but we need (b, heads, c_head, h*w)
        q = q.view(b, self.num_heads, c_head, h * w)
        k = k.view(b, self.num_heads, c_head, h * w)
        v = v.view(b, self.num_heads, c_head, h * w)
        # Normalize q and k along the last dimension (tokens) as in the original MDTA
        # Compute the magnitude squared across real and imaginary parts
        q_mag = (q.real**2 + q.imag**2) + 1e-6
        k_mag = (k.real**2 + k.imag**2) + 1e-6
        # Normalize across the token dimension
        q_norm = torch.linalg.norm(q_mag, dim=-1, keepdim=True)
        k_norm = torch.linalg.norm(k_mag, dim=-1, keepdim=True)
        q = q / (q_norm + 1e-6)
        k = k / (k_norm + 1e-6)
        # Compute complex dot product of q and conjugate of k along the token dimension
        # The real part of the dot product is used to build the attention scores
        # conj(k) = k.real - i k.imag; real part of q * conj(k) is (q.real*k.real + q.imag*k.imag)
        # We perform matrix multiplication along the token dimension (-1)
        real_scores = torch.matmul(q.real, k.real.transpose(-2, -1)) + torch.matmul(q.imag, k.imag.transpose(-2, -1))
        # Apply temperature and softmax to obtain attention weights
        attn = torch.softmax(real_scores * self.temperature, dim=-1)
        # Weighted sum of values
        # v has shape (b, heads, c_head, h*w).  attn: (b, heads, c_head, c_head)
        out_real = torch.matmul(attn, v.real)
        out_imag = torch.matmul(attn, v.imag)
        out = out_real + 1j * out_imag
        # Reshape back to (b, c, h, w)
        out = out.view(b, c, h, w)
        # Final projection
        out = self.project_out(out)
        return out


class ComplexGDFN(nn.Module):
    """Complex gated depthwise feed–forward network (GDFN).

    This is a complex variant of the GDFN block described in the
    Restormer implementation 【802216581475939†L31-L47】.  It uses complex
    convolutions to project the input to a higher dimensional space,
    applies a depthwise convolution and gating via a complex GELU, and
    then projects back to the original channel dimension.
    """

    def __init__(self, channels: int, expansion_factor: float) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        # Project input to 2 × hidden channels
        self.project_in = ComplexConv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        # Depthwise convolution on the projected tensor
        self.conv = ComplexConv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        # Project back to original channel dimension
        self.project_out = ComplexConv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexGDFN expects a complex input tensor, got dtype {x.dtype}"
            )
        # Project and split into two halves
        x_proj = self.conv(self.project_in(x))
        x1, x2 = torch.chunk(x_proj, chunks=2, dim=1)
        # Apply complex GELU to the first half
        x1 = complex_gelu(x1)
        # Element‐wise complex multiplication with the second half
        x_combined = x1 * x2
        # Project back to the original channel dimension
        return self.project_out(x_combined)

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
