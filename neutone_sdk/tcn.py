"""
Code based off https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb
"""

import logging
import os
from typing import Optional, Callable, List

import torch as tr
from torch import Tensor
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def center_crop(x: Tensor, length: int) -> Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        start = (x.size(-1) - length) // 2
        stop = start + length
        x = x[..., start:stop]
    return x


def causal_crop(x: Tensor, length: int) -> Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        stop = x.size(-1) - 1
        start = stop - length
        x = x[..., start:stop]
    return x


class FiLM(nn.Module):
    def __init__(self,
                 cond_dim: int,  # dim of conditioning input
                 num_features: int,  # dim of the conv channel
                 use_bn: bool = True) -> None:  # TODO(cm): check what this default value should be
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g[:, :, None]
        b = b[:, :, None]
        if self.use_bn:
            x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine
        return x


class TCNBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1,
                 padding: Optional[int] = 0,
                 use_ln: bool = False,
                 temporal_dim: Optional[int] = None,
                 use_act: bool = True,
                 use_res: bool = True,
                 cond_dim: int = 0,
                 use_film_bn: bool = True,
                 crop_fn: Callable[[Tensor, int], Tensor] = causal_crop) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.use_ln = use_ln
        self.temporal_dim = temporal_dim
        self.use_act = use_act
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.crop_fn = crop_fn

        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
            log.debug(f"Setting padding automatically to {padding} samples")
        self.padding = padding

        self.ln = None
        if use_ln:
            assert temporal_dim is not None and temporal_dim > 0
            self.ln = nn.LayerNorm([in_ch, temporal_dim], elementwise_affine=False)

        self.act = None
        if use_act:
            self.act = nn.PReLU(out_ch)

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            (kernel_size,),
            stride=(stride,),
            padding=padding,
            dilation=(dilation,),
            bias=True,
        )
        self.res = None
        if use_res:
            self.res = nn.Conv1d(in_ch, out_ch, kernel_size=(1,), stride=(stride,), bias=False)

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_ch, use_bn=use_film_bn)

    def is_conditional(self) -> bool:
        return self.cond_dim > 0

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x_in = x
        if self.ln is not None:
            assert x.size(1) == self.in_ch
            assert x.size(2) == self.temporal_dim
            x = self.ln(x)
        x = self.conv(x)
        if self.is_conditional():
            assert cond is not None
            x = self.film(x, cond)
        if self.act is not None:
            x = self.act(x)
        if self.res is not None:
            res = self.res(x_in)
            x_res = self.crop_fn(res, x.size(-1))
            x += x_res
        return x


class TCN(nn.Module):
    def __init__(self,
                 out_channels: List[int],
                 dilations: Optional[List[int]] = None,
                 in_ch: int = 1,
                 kernel_size: int = 13,
                 strides: Optional[List[int]] = None,
                 padding: Optional[int] = 0,
                 use_ln: bool = False,
                 temporal_dims: Optional[List[int]] = None,
                 use_act: bool = True,
                 use_res: bool = True,
                 cond_dim: int = 0,
                 use_film_bn: bool = False,
                 crop_fn: Callable[[Tensor, int], Tensor] = causal_crop) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_ch = in_ch
        self.out_ch = out_channels[-1]
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_ln = use_ln
        self.temporal_dims = temporal_dims  # TODO(cm): calculate automatically
        self.use_act = use_act
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.crop_fn = crop_fn
        # TODO(cm): padding warning

        self.n_blocks = len(out_channels)
        if dilations is None:
            dilations = [4 ** idx for idx in range(self.n_blocks)]
            log.info(f"Setting dilations automatically to: {dilations}")
        assert len(dilations) == self.n_blocks
        self.dilations = dilations

        if strides is None:
            strides = [1] * self.n_blocks
            log.info(f"Setting strides automatically to: {strides}")
        assert len(strides) == self.n_blocks
        self.strides = strides

        if use_ln:
            assert temporal_dims is not None
            assert len(temporal_dims) == self.n_blocks

        self.blocks = nn.ModuleList()
        block_out_ch = None
        for idx, (curr_out_ch, dil, stride) in enumerate(zip(out_channels, dilations, strides)):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            temp_dim = None
            if temporal_dims is not None:
                temp_dim = temporal_dims[idx]

            self.blocks.append(TCNBlock(
                block_in_ch,
                block_out_ch,
                kernel_size,
                dil,
                stride,
                padding,
                use_ln,
                temp_dim,
                use_act,
                use_res,
                cond_dim,
                use_film_bn,
                crop_fn
            ))

    def is_conditional(self) -> bool:
        return self.cond_dim > 0

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if self.is_conditional():
            assert cond is not None
            assert cond.shape == (x.size(0), self.cond_dim)  # (batch_size, cond_dim)
        for block in self.blocks:
            x = block(x, cond)
        return x

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples."""
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf
