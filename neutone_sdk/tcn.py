import logging
import os
from typing import Optional, List

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


# TODO(cm): optimize for TorchScript
class PaddingCached(nn.Module):
    """Cached padding for cached convolutions."""
    def __init__(self, n_ch: int, padding: int) -> None:
        super().__init__()
        self.n_ch = n_ch
        self.padding = padding
        self.register_buffer("pad_buf", tr.zeros((1, n_ch, padding)))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        bs = x.size(0)
        if bs > self.pad_buf.size(0):  # Perform resizing once if batch size is not 1
            self.pad_buf = self.pad_buf.repeat(bs, 1, 1)
        x = tr.cat([self.pad_buf, x], dim=-1)  # concat input signal to the cache
        self.pad_buf = x[..., -self.padding:]  # discard old cache
        return x


class Conv1dCached(nn.Module):  # Conv1d with cache
    """Cached causal convolution for streaming."""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        assert padding == 0  # We include padding in the constructor to match the Conv1d constructor
        padding = (kernel_size - 1) * dilation
        self.pad = PaddingCached(in_channels, padding)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              (kernel_size,),
                              (stride,),
                              padding=0,
                              dilation=(dilation,),
                              bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        return x


class FiLM(nn.Module):
    def __init__(self,
                 cond_dim: int,  # dim of conditioning input
                 num_features: int,  # dim of the conv channel
                 use_bn: bool = True) -> None:  # TODO(cm): check what this default value should be
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        self.bn = None
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        if self.bn is not None:
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
                 is_causal: bool = True,
                 is_cached: bool = False) -> None:
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
        self.is_causal = is_causal
        self.is_cached = is_cached
        if is_causal:
            assert padding == 0, "If the TCN is causal, padding must be 0"
            self.crop_fn = causal_crop
        else:
            self.crop_fn = center_crop
        if is_cached:
            assert is_causal, "If the TCN is streaming, it must be causal"
            self.conv_cls = Conv1dCached
        else:
            self.conv_cls = nn.Conv1d

        if padding is None:
            padding = kernel_size // 2 * dilation
            log.debug(f"Setting padding automatically to {padding} samples")
        self.padding = padding

        self.ln = None
        if use_ln:
            assert temporal_dim is not None and temporal_dim > 0
            self.ln = nn.LayerNorm([in_ch, temporal_dim], elementwise_affine=False)

        self.act = None
        if use_act:
            self.act = nn.PReLU(out_ch)

        self.conv = self.conv_cls(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
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
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        x_in = x
        if self.ln is not None:
            assert x.size(1) == self.in_ch
            assert x.size(2) == self.temporal_dim
            x = self.ln(x)
        x = self.conv(x)
        if self.film is not None:
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
                 is_causal: bool = True,
                 is_cached: bool = False) -> None:
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
        self.is_causal = is_causal
        self.is_cached = is_cached
        if is_causal:
            assert padding == 0, "If the TCN is causal, padding must be 0"
            self.crop_fn = causal_crop
        else:
            self.crop_fn = center_crop
        if is_cached:
            assert is_causal, "If the TCN is streaming, it must be causal"

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
                is_causal,
                is_cached
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


if __name__ == '__main__':
    out_channels = [8] * 4
    tcn = TCN(out_channels, cond_dim=3, padding=0, is_causal=True, is_cached=True)
    log.info(tcn.calc_receptive_field())
    audio = tr.rand((1, 1, 65536))
    cond = tr.rand((1, 3))
    # cond = None
    out = tcn.forward(audio, cond)
    log.info(out.shape)
