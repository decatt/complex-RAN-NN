"""
complex_mimo_unet.py
====================

This module implements a complex‑valued variant of the MIMO‑UNet
architecture originally proposed for single image deblurring in
`Rethinking Coarse‑to‑Fine Approach in Single Image Deblurring` (ICCV 2021)
【257032074227535†L351-L364】.  The real‐valued MIMO‑UNet builds upon a
U‑shaped encoder–decoder network but introduces three key ideas: a
multi‑input single encoder (MISE), a multi‑output single decoder (MOSD)
and asymmetric feature fusion (AFF) to efficiently propagate information
across scales.  Here we provide a simplified yet flexible complex
implementation suitable for processing complex‑valued inputs such as
synthetic aperture radar (SAR) images or other signals where both
amplitude and phase information are important.

The building blocks in this file make heavy use of the
``complex_layer`` library (shipped in this repository) which supplies
complex convolution, activation and normalization layers.  All
convolutions operate directly on complex tensors (dtype
``torch.complex64`` or ``torch.complex128``) without manually splitting
the real and imaginary parts.  To support multi‑scale processing we
provide simple helper functions for downsampling and upsampling
complex tensors by applying the corresponding real‑valued operation to
the real and imaginary components separately.  These helpers avoid
PyTorch operators that currently lack native complex support.

The main classes defined here are:

* ``ComplexMIMOUnetBlock`` – a two‑layer convolutional block with
  complex normalization and activation.  It mirrors the basic
  conv–norm–activation pattern commonly used in UNet architectures.

* ``ComplexMIMOUnet`` – a three‑level encoder–decoder network that
  ingests a complex tensor of shape ``(B, C, H, W)`` and returns a
  tensor of identical shape.  The network accepts multi‑scale inputs
  (coarse versions of the input generated on the fly) and fuses these
  features into the encoder.  Skip connections and residual addition
  ensure that both low‑level and high‑level information contribute to
  the final output.

This implementation is intentionally modular so that the number of
levels and base channel widths may be adjusted by the caller.  While
the original MIMO‑UNet outputs multiple deblurred images at
intermediate scales 【257032074227535†L351-L364】, the provided model
produces a single high‑resolution output for simplicity.  Additional
heads could be attached at various decoder stages to recover
multi‑scale outputs if required.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from complex_layer import (
    complex_gelu,
    ComplexConv2d,
    ComplexLayerNorm,
)


def complex_avg_pool2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Downsample a complex tensor using average pooling.

    PyTorch currently does not support pooling operations on complex
    tensors directly.  This helper computes the average pool of the
    real and imaginary parts separately and recombines them into a
    complex result.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor of shape ``(B, C, H, W)``.
    kernel_size : int
        Size of the pooling window.  The stride is set equal to
        ``kernel_size`` so that the spatial dimensions are reduced by
        exactly this factor.

    Returns
    -------
    torch.Tensor
        Downsampled complex tensor.  The channel dimension is left
        unchanged but height and width are divided by ``kernel_size``.
    """
    if not x.is_complex():
        raise TypeError(f"complex_avg_pool2d expects a complex tensor, got {x.dtype}")
    # Apply average pooling independently to real and imaginary parts.
    real = F.avg_pool2d(x.real, kernel_size, stride=kernel_size)
    imag = F.avg_pool2d(x.imag, kernel_size, stride=kernel_size)
    return real.to(x.dtype) + 1j * imag.to(x.dtype)


def complex_interpolate(
    x: torch.Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Upsample a complex tensor by interpolating the real and imaginary parts.

    This helper wraps :func:`torch.nn.functional.interpolate` so that
    complex inputs are supported by applying interpolation to the
    components separately.  Either ``size`` or ``scale_factor`` must
    be specified.  See the PyTorch documentation for further details
    on the interpolation options.

    Parameters
    ----------
    x : torch.Tensor
        Complex tensor of shape ``(B, C, H, W)``.
    size : list[int], optional
        Target spatial size ``[H_out, W_out]``.  Exactly one of
        ``size`` or ``scale_factor`` must be given.
    scale_factor : float, optional
        Factor by which to multiply the spatial dimensions.  For
        example, a value of 2.0 doubles height and width.
    mode : str, default "bilinear"
        Interpolation algorithm.  Bilinear interpolation is used by
        default, but other modes (e.g. ``nearest``) are valid.
    align_corners : bool, default False
        Passed through to :func:`torch.nn.functional.interpolate`.

    Returns
    -------
    torch.Tensor
        Upsampled complex tensor.
    """
    if not x.is_complex():
        raise TypeError(f"complex_interpolate expects a complex tensor, got {x.dtype}")
    if size is None and scale_factor is None:
        raise ValueError("Either 'size' or 'scale_factor' must be specified")
    real = F.interpolate(x.real, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    imag = F.interpolate(x.imag, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    return real.to(x.dtype) + 1j * imag.to(x.dtype)


class ComplexMIMOUnetBlock(nn.Module):
    """A two‑layer convolutional block for complex inputs.

    Each block consists of two complex convolution layers interleaved
    with optional complex layer normalization and a pointwise
    activation.  The design follows the conv–norm–activation pattern
    common to real‑valued UNet implementations, adapted to the
    complex domain using the layers provided by :mod:`complex_layer`.

    Parameters
    ----------
    in_channels : int
        Number of complex channels in the input.
    out_channels : int
        Number of complex channels produced by this block.
    norm : bool, default True
        Whether to apply complex layer normalization after each
        convolution.
    activation : callable, default :func:`complex_gelu`
        Pointwise activation function for complex inputs.  The
        default applies the GELU nonlinearity separately to the
        real and imaginary parts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        norm: bool = True,
        activation: callable = complex_gelu,
    ) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = ComplexLayerNorm(out_channels) if norm else nn.Identity()
        self.act1 = activation
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = ComplexLayerNorm(out_channels) if norm else nn.Identity()
        self.act2 = activation

    def _apply_norm(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        """Apply complex layer normalization to a 4D tensor.

        The :class:`ComplexLayerNorm` provided by ``complex_layer`` expects
        inputs of shape ``(B, N, C)`` where the last dimension
        corresponds to the channels.  This helper reshapes a tensor
        of shape ``(B, C, H, W)`` into the expected format, applies
        normalization and reshapes the result back to the original
        dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input complex tensor of shape ``(B, C, H, W)``.
        norm : nn.Module
            Instance of :class:`ComplexLayerNorm` or
            :class:`torch.nn.Identity`.

        Returns
        -------
        torch.Tensor
            Normalized complex tensor with the same shape as ``x``.
        """
        if isinstance(norm, nn.Identity):
            return x
        b, c, h, w = x.shape
        # Reshape to (B, H*W, C) for layernorm
        x_reshaped = x.view(b, c, h * w).transpose(-2, -1)  # (B, H*W, C)
        x_norm = norm(x_reshaped)
        # Reshape back to (B, C, H, W)
        x_norm = x_norm.transpose(-2, -1).view(b, c, h, w)
        return x_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            raise TypeError(f"ComplexMIMOUnetBlock expects a complex input tensor, got {x.dtype}")
        # First convolution
        x = self.conv1(x)
        x = self._apply_norm(x, self.norm1)
        x = self.act1(x)
        # Second convolution
        x = self.conv2(x)
        x = self._apply_norm(x, self.norm2)
        x = self.act2(x)
        return x


class ComplexMIMOUnet(nn.Module):
    """A three‑level complex MIMO‑UNet.

    The network accepts a complex tensor of shape ``(B, C, H, W)`` and
    produces an output of the same shape.  It synthesizes features
    across three spatial scales by downsampling the input image on the
    fly and feeding those coarse representations into deeper encoder
    levels.  Skip connections and upsampling mirror the structure of
    standard UNet architectures.  A final residual addition merges the
    learned correction with the original input.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input (e.g. 3 for RGB).  The output
        will have the same number of channels.
    base_channels : int, default 64
        Width of the network.  Deeper levels allocate multiples of
        ``base_channels`` channels.
    num_scales : int, default 3
        Number of downsampled copies of the input to fuse into the
        encoder.  When set to 3, three scales are used: the original
        resolution, half resolution and quarter resolution.  Additional
        scales can be specified, but care must be taken to adjust the
        fusion and projection layers accordingly.
    norm : bool, default True
        Whether to apply complex layer normalization in each
        convolutional block.
    activation : callable, default :func:`complex_gelu`
        Activation function applied after each convolution.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_scales: int = 3,
        *,
        norm: bool = True,
        activation: callable = complex_gelu,
    ) -> None:
        super().__init__()
        assert num_scales >= 1, "num_scales must be at least 1"
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_scales = num_scales

        # Shallow feature extraction (SCM) for each input scale.  Each
        # scale projection maps the raw input to a base number of channels.
        self.scm = nn.ModuleList([
            ComplexConv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(num_scales)
        ])

        # Projection layers to match scale features to encoder channel widths.
        # These 1×1 convolutions map the shallow multi‑scale features
        # to the number of channels expected by each encoder level.  The
        # projections are ordered from finest to coarsest scale.  When
        # ``num_scales`` is greater than 3, additional projection layers
        # should be appended to handle deeper levels.
        self.proj_l1 = ComplexConv2d(base_channels, base_channels * 4, kernel_size=1, bias=False)
        self.proj_l2 = ComplexConv2d(base_channels, base_channels * 8, kernel_size=1, bias=False)

        # Encoder blocks.  The first encoder doubles the channel count.
        self.enc1 = ComplexMIMOUnetBlock(base_channels, base_channels * 2, norm=norm, activation=activation)
        # Downsample modules use the complex downsample from complex_layer.
        # At each downsampling step channels double: see complex_layer.ComplexDownSample.
        from complex_layer import ComplexDownSample, ComplexUpSample  # imported here to avoid circular imports

        self.down1 = ComplexDownSample(base_channels * 2)
        self.enc2 = ComplexMIMOUnetBlock(base_channels * 4, base_channels * 4, norm=norm, activation=activation)
        self.down2 = ComplexDownSample(base_channels * 4)
        self.enc3 = ComplexMIMOUnetBlock(base_channels * 8, base_channels * 8, norm=norm, activation=activation)

        # Bottleneck block (deepest level)
        self.bottom = ComplexMIMOUnetBlock(base_channels * 8, base_channels * 8, norm=norm, activation=activation)

        # Decoder blocks and upsamplers.  Upward path halves the
        # channel count while doubling spatial dimensions.
        self.up2 = ComplexUpSample(base_channels * 8)
        self.dec2 = ComplexMIMOUnetBlock(base_channels * 8, base_channels * 4, norm=norm, activation=activation)
        self.up1 = ComplexUpSample(base_channels * 4)
        self.dec1 = ComplexMIMOUnetBlock(base_channels * 4, base_channels * 2, norm=norm, activation=activation)

        # Final output projection to match the number of input channels.
        self.final_conv = ComplexConv2d(base_channels * 2, in_channels, kernel_size=3, padding=1, bias=False)

    def _prepare_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Generate a list of downsampled inputs for multi‑scale processing.

        The returned list always contains the original input ``x`` as its
        first element.  Subsequent elements are downsampled by factors
        of 2 using average pooling on the real and imaginary parts.
        This helper is independent of ``num_scales``; the caller
        should slice the list to obtain the desired number of scales.

        Parameters
        ----------
        x : torch.Tensor
            Complex tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        list[torch.Tensor]
            List containing ``num_scales`` complex tensors with
            progressively reduced spatial resolutions.
        """
        scales = [x]
        current = x
        # Generate additional scales by repeatedly downsampling by 2.
        for _ in range(1, self.num_scales):
            current = complex_avg_pool2d(current, kernel_size=2)
            scales.append(current)
        return scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the complex MIMO‑UNet.

        The method performs the following high‑level steps:

        1.  Generate multi‑scale inputs using average pooling on
            complex tensors.
        2.  Project each scale to a base number of channels via the
            ``scm`` layers.
        3.  Upsample coarse features to the finest scale and sum them
            (with broadcasting) to form the input to the first encoder.
        4.  Encode the fused features through a series of complex
            convolutional blocks and downsample operations while
            injecting projected coarse features into deeper levels.
        5.  Decode the deepest representation using upsampling and
            convolutional blocks with skip connections.
        6.  Apply a final convolution and add a residual connection to
            produce the output.

        Parameters
        ----------
        x : torch.Tensor
            Input complex tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Output complex tensor of shape ``(B, C, H, W)``.
        """
        if not x.is_complex():
            raise TypeError(f"ComplexMIMOUnet expects a complex input tensor, got {x.dtype}")

        # Prepare multi‑scale inputs (e.g. x, downsample(x), downsample(x/2), ...)
        scales = self._prepare_multi_scale(x)
        # Extract shallow features for each scale
        shallow_feats = [scm_layer(s) for scm_layer, s in zip(self.scm, scales)]

        # Determine spatial sizes for upsampling coarse features to the finest resolution
        _, _, h, w = x.shape
        # Accumulate features from all scales at the finest resolution
        fused_fine = shallow_feats[0]
        for lvl in range(1, self.num_scales):
            s_feat = shallow_feats[lvl]
            # Upsample to the size of the original image
            s_up = complex_interpolate(s_feat, size=[h, w], mode="bilinear", align_corners=False)
            fused_fine = fused_fine + s_up

        # Encoder level 1
        enc1 = self.enc1(fused_fine)  # (B, base*2, H, W)
        # Downsample enc1
        down1 = self.down1(enc1)      # (B, base*4, H/2, W/2)

        # Prepare and inject scale 2 features into level 2
        if self.num_scales > 1:
            # shallow_feats[1] has shape (B, base, H/2, W/2).  Project it to base*4
            s2_proj = self.proj_l1(shallow_feats[1])
            # Add projected features to the downsampled encoder features
            down1 = down1 + s2_proj

        # Encoder level 2
        enc2 = self.enc2(down1)       # (B, base*4, H/2, W/2)
        # Downsample enc2
        down2 = self.down2(enc2)      # (B, base*8, H/4, W/4)

        # Prepare and inject scale 3 features into level 3
        if self.num_scales > 2:
            # shallow_feats[2] has shape (B, base, H/4, W/4).  Project it to base*8
            s3_proj = self.proj_l2(shallow_feats[2])
            down2 = down2 + s3_proj

        # Encoder level 3 (bottleneck)
        enc3 = self.enc3(down2)       # (B, base*8, H/4, W/4)
        bottom = self.bottom(enc3)    # (B, base*8, H/4, W/4)

        # Decoder level 2
        up2 = self.up2(bottom)        # (B, base*4, H/2, W/2)
        # Concatenate skip connection from enc2
        cat2 = torch.cat([up2, enc2], dim=1)  # (B, base*8, H/2, W/2)
        dec2 = self.dec2(cat2)        # (B, base*4, H/2, W/2)

        # Decoder level 1
        up1 = self.up1(dec2)          # (B, base*2, H, W)
        cat1 = torch.cat([up1, enc1], dim=1)  # (B, base*4, H, W)
        dec1 = self.dec1(cat1)        # (B, base*2, H, W)

        out = self.final_conv(dec1)    # (B, in_channels, H, W)
        # Residual connection
        out = out + x
        return out

if __name__ == "__main__":
    # Create a dummy complex input tensor (batch=2, channels=3, height=64, width=64)
    x = torch.randn(48, 34, 32, 96, dtype=torch.complex64)

    # Instantiate the ComplexMIMOUnet model
    model = ComplexMIMOUnet(in_channels=34)

    # Forward pass
    y = model(x)

    # Print output shape
    print(y.shape)  # Expected output: (48, 34, 32, 96)
