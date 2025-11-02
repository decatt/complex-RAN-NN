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
implementation.  Key building blocks such as
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
  Restormer paper.

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

from complex_layer import complex_gelu, ComplexConv2d, ComplexLayerNorm, ComplexPixelShuffle, ComplexPixelUnshuffle, ComplexDownSample, ComplexUpSample, ComplexMDTA, ComplexGDFN  # type: ignore

__all__ = [
    "ComplexRestormerBlock",
    "ComplexRestormer",
]

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
