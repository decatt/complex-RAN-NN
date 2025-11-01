"""
complex_restormer.py
====================

This module implements a complex‐valued variant of the Restormer
architecture for image restoration.  It is built on top of the
``complex_layer`` library which provides basic complex convolution and
linear layers.  The goal of ``ComplexRestormer`` is to take a
complex‐valued image tensor of shape ``(B, C, H, W)`` and return a
tensor of the same shape, making it suitable for tasks like denoising
or deblurring in the complex domain.

The implementation draws inspiration from the real‐valued Restormer
implementation 【802216581475939†L4-L29】.  Key building blocks such as
multi–DConv head transposed self‐attention (MDTA) and gated depthwise
feed–forward networks (GDFN) have been adapted to operate on complex
numbers.  Complex normalization is performed by applying
``nn.LayerNorm`` independently to the real and imaginary parts.  The
downsampling and upsampling modules employ pixel unshuffle/shuffle
operations on both real and imaginary components.  This design mirrors
the structure of the original Restormer while respecting the algebra
of complex numbers.

Classes defined in this file:

* ``ComplexLayerNorm`` – performs layer normalization on complex inputs by
  normalizing the real and imaginary parts separately.
* ``ComplexPixelShuffle`` / ``ComplexPixelUnshuffle`` – wrappers around
  PyTorch’s ``PixelShuffle`` and ``PixelUnshuffle`` for complex data.
* ``ComplexDownSample`` / ``ComplexUpSample`` – downsample or upsample
  spatial dimensions using complex convolutions and pixel (un)shuffle.
* ``ComplexMDTA`` – complex analogue of the MDTA block used in Restormer;
  it computes attention weights from complex queries and keys and
  applies them to complex values.
* ``ComplexGDFN`` – complex gated depthwise feed–forward network.
* ``ComplexRestormerBlock`` – the basic transformer block combining
  normalization, complex MDTA and complex GDFN with residual
  connections.
* ``ComplexRestormer`` – the full encoder–decoder network with skip
  connections and refinement stage as described in the original
  Restormer paper 【802216581475939†L88-L149】.

Usage::

    import torch
    from complex_layer import ComplexConv2d  # ensure the complex layer file is in your path
    from complex_restormer import ComplexRestormer

    # create a dummy complex input tensor (batch=2, channels=3, height=64, width=64)
    x = torch.randn(2, 3, 64, 64, dtype=torch.complex64)
    model = ComplexRestormer(in_channels=3)
    y = model(x)
    print(y.shape)  # (2, 3, 64, 64)

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from complex_layer import ComplexConv2d  # type: ignore

__all__ = [
    "ComplexLayerNorm",
    "ComplexPixelShuffle",
    "ComplexPixelUnshuffle",
    "ComplexDownSample",
    "ComplexUpSample",
    "ComplexMDTA",
    "ComplexGDFN",
    "ComplexRestormerBlock",
    "ComplexRestormer",
]


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
    the original Restormer 【802216581475939†L69-L76】.
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
    the original Restormer 【802216581475939†L79-L86】.
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


class ComplexRestormerBlock(nn.Module):
    """A basic transformer block for complex inputs.

    This block combines complex layer normalization, complex MDTA and
    complex GDFN with residual connections.  It mirrors the structure
    of the real‐valued ``TransformerBlock`` in Restormer 【802216581475939†L49-L66】.
    """

    def __init__(self, channels: int, num_heads: int, expansion_factor: float) -> None:
        super().__init__()
        self.norm1 = ComplexLayerNorm(channels)
        self.attn = ComplexMDTA(channels, num_heads)
        self.norm2 = ComplexLayerNorm(channels)
        self.ffn = ComplexGDFN(channels, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexRestormerBlock expects a complex input tensor, got dtype {x.dtype}"
            )
        b, c, h, w = x.shape
        # Normalize across channel dimension for each pixel
        x_reshaped = x.reshape(b, c, -1).transpose(-2, -1)  # (b, h*w, c)
        x_norm = self.norm1(x_reshaped)
        x_norm = x_norm.transpose(-2, -1).reshape(b, c, h, w)
        x = x + self.attn(x_norm)
        # Second normalization and feed‐forward
        x_reshaped = x.reshape(b, c, -1).transpose(-2, -1)  # (b, h*w, c)
        x_norm = self.norm2(x_reshaped)
        x_norm = x_norm.transpose(-2, -1).reshape(b, c, h, w)
        x = x + self.ffn(x_norm)
        return x


class ComplexRestormer(nn.Module):
    """The complex Restormer network.

    This is an encoder–decoder architecture with skip connections and a
    refinement stage.  It follows the design of the real Restormer
    described in the original implementation 【802216581475939†L88-L149】 but
    operates entirely in the complex domain using the modules defined
    above.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input tensor (e.g. 3 for RGB).  The
        output will have the same number of channels.
    num_blocks : List[int], optional
        Number of ``ComplexRestormerBlock``s in each encoder stage.
    num_heads : List[int], optional
        Number of attention heads in each encoder stage.
    channels : List[int], optional
        Base channel widths for each encoder stage.
    num_refinement : int, optional
        Number of refinement blocks applied after decoding.
    expansion_factor : float, optional
        Expansion factor used in the GDFN blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_blocks: List[int] = [4, 6, 6, 8],
        num_heads: List[int] = [1, 2, 4, 8],
        channels: List[int] = [48, 96, 192, 384],
        num_refinement: int = 4,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()
        assert len(num_blocks) == len(num_heads) == len(channels), "num_blocks, num_heads and channels must have the same length"
        self.in_channels = in_channels
        # Initial embedding convolution
        self.embed_conv = ComplexConv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        # Encoder stages
        self.encoders = nn.ModuleList([
            nn.Sequential(*[ComplexRestormerBlock(num_ch, num_ah, expansion_factor) for _ in range(num_tb)])
            for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
        ])
        # Downsample modules (one less than the number of stages)
        self.downs = nn.ModuleList([ComplexDownSample(num_ch) for num_ch in channels[:-1]])
        # Upsample modules for decoding (reverse channels order except the last)
        self.ups = nn.ModuleList([ComplexUpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # Reduce convolutions to merge skip connections
        # There are (len(channels) - 2) reduce convolutions
        self.reduces = nn.ModuleList([
            ComplexConv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
            for i in reversed(range(2, len(channels)))
        ])
        # Decoder stages: one fewer than encoders
        # The first decoder operates on channels[2], the second on channels[1], and the last on channels[1]
        self.decoders = nn.ModuleList([
            nn.Sequential(*[ComplexRestormerBlock(channels[2], num_heads[2], expansion_factor) for _ in range(num_blocks[2])]),
            nn.Sequential(*[ComplexRestormerBlock(channels[1], num_heads[1], expansion_factor) for _ in range(num_blocks[1])]),
            nn.Sequential(*[ComplexRestormerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_blocks[0])]),
        ])
        # Refinement stage
        self.refinement = nn.Sequential(*[
            ComplexRestormerBlock(channels[1], num_heads[0], expansion_factor) for _ in range(num_refinement)
        ])
        # Final output convolution
        self.output = ComplexConv2d(channels[1], in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(
                f"ComplexRestormer expects a complex input tensor, got dtype {x.dtype}"
            )
        b, c_in, h, w = x.shape
        # Embedding
        fo = self.embed_conv(x)
        # Encoder path
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))
        # Decoder path
        # First decoder stage: upsample and combine with encoder 3, then reduce channels
        up3 = self.ups[0](out_enc4)
        cat3 = torch.cat([up3, out_enc3], dim=1)
        dec3 = self.decoders[0](self.reduces[0](cat3))
        # Second decoder stage: upsample and combine with encoder 2
        up2 = self.ups[1](dec3)
        cat2 = torch.cat([up2, out_enc2], dim=1)
        dec2 = self.decoders[1](self.reduces[1](cat2))
        # Third decoder stage: upsample and combine with encoder 1
        up1 = self.ups[2](dec2)
        cat1 = torch.cat([up1, out_enc1], dim=1)
        dec1 = self.decoders[2](cat1)
        # Refinement
        fr = self.refinement(dec1)
        # Output
        out = self.output(fr) + x
        return out

if __name__ == "__main__":
    # Create a dummy complex input tensor (batch=2, channels=3, height=64, width=64)
    x = torch.randn(2, 3, 64, 64, dtype=torch.complex64)

    # Instantiate the ComplexRestormer model
    model = ComplexRestormer(in_channels=3)

    # Forward pass
    y = model(x)

    # Print output shape
    print(y.shape)  # Expected output: (2, 3, 64, 64)
