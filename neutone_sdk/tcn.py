import logging
import os
from typing import Optional, List, Tuple, Union
import torch.nn.functional as F

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


class PaddingCached(nn.Module):
    def __init__(self,
                 n_ch: int,
                 padding: int,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 debug_mode: bool = True) -> None:
        """
        Cached padding for cached convolutions. Handles dynamic batch sizes by default
        at the expense of dynamic memory allocations.

        Args:
            n_ch: Number of channels.
            padding: Number of padding samples.
            use_dynamic_bs: If True, the padding will dynamically change batch size to
                            match the input.
            batch_size: If known, the initial batch size can be specified here to avoid
                        dynamic changes.
            debug_mode: If True, assert statements are enabled.
        """
        super().__init__()
        if debug_mode:
            assert n_ch > 0
            assert batch_size > 0
        self.n_ch = n_ch
        self.padding = padding
        self.use_dynamic_bs = use_dynamic_bs
        self.debug_mode = debug_mode
        self.register_buffer("pad_buf", tr.zeros((batch_size, n_ch, padding)))

    def reset(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None:
            self.pad_buf = self.pad_buf.new_zeros((batch_size, self.n_ch, self.padding))
        else:
            self.pad_buf.zero_()

    def prepare_for_inference(self) -> None:
        self.debug_mode = False
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        if self.debug_mode:
            assert x.ndim == 3  # (batch_size, in_ch, samples)
        # We support padding == 0 for convolutions with kernel size of 1
        if self.padding == 0:
            return x

        bs = x.size(0)
        if self.use_dynamic_bs and bs != self.pad_buf.size(0):
            self.reset(bs)
        elif self.debug_mode:
            assert bs == self.pad_buf.size(0)

        x = tr.cat([self.pad_buf, x], dim=-1)  # Concat input to the cache
        self.pad_buf = x[..., -self.padding:]  # Discard old cache
        return x


class Conv1dGeneral(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: Union[int, Tuple[int, int], str] = "same",
                 dilation: int = 1,
                 bias: bool = True,
                 padding_mode: str = "zeros",
                 causal: bool = True,
                 cached: bool = False,
                 use_dynamic_bs: bool = True,
                 batch_size: int = 1,
                 debug_mode: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.padding_mode = padding_mode
        self.causal = causal
        self.cached = cached
        self.debug_mode = debug_mode

        left_padding, right_padding = self._calc_padding(kernel_size,
                                                         stride,
                                                         padding,
                                                         dilation,
                                                         causal)
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.uncached_padding = (left_padding, right_padding)

        self.conv1d = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size=(kernel_size,),
                                stride=(stride,),
                                padding=0,
                                dilation=(dilation,),
                                bias=bias,
                                padding_mode=padding_mode)
        self.padding_cached = PaddingCached(in_channels,
                                            left_padding,
                                            use_dynamic_bs,
                                            batch_size,
                                            debug_mode)

    def _calc_padding(self,
                      kernel_size: int,
                      stride: int,
                      padding: Union[int, Tuple[int, int], str],
                      dilation: int,
                      causal: bool) -> Tuple[int, int]:
        if padding == "valid":
            return 0, 0
        elif padding == "same":
            assert stride == 1, "If padding is 'same', stride must be 1"
            pad_amount = (kernel_size - 1) * dilation
            if causal:
                return pad_amount, 0
            elif pad_amount % 2 == 0:
                return pad_amount // 2, pad_amount // 2
            else:
                # Favor left padding over right padding if the padding amount is odd
                return pad_amount // 2 + 1, pad_amount // 2
        elif isinstance(padding, int):
            assert padding >= 0
            if causal:
                return padding, 0
            else:
                return padding, padding
        else:
            assert len(padding) == 2, "Expected padding to be a tuple of length 2."
            assert padding[0] >= 0 and padding[1] >= 0
            if causal:
                assert padding[1] == 0, "If causal, right padding must be 0"
            return padding

    def set_cached(self, cached: bool) -> None:
        self.cached = cached
        self.reset()

    def reset(self, batch_size: Optional[int] = None) -> None:
        self.padding_cached.reset(batch_size)

    def prepare_for_inference(self) -> None:
        self.debug_mode = False
        self.conv1d.eval()
        self.padding_cached.prepare_for_inference()

    def forward(self, x: Tensor) -> Tensor:
        if self.debug_mode:
            assert x.ndim == 3  # (batch_size, in_ch, samples)
            assert x.size(1) == self.in_channels
        if self.cached:
            x = self.padding_cached(x)
            if self.right_padding > 0:
                # TODO(cm): prevent dynamic memory allocations here
                x = F.pad(x, (0, self.right_padding), mode=self.padding_mode)
        elif self.uncached_padding != (0, 0):
            # TODO(cm): prevent dynamic memory allocations here
            x = F.pad(x, self.uncached_padding, mode=self.padding_mode)
        x = self.conv1d(x)
        return x


class Conv1dCached(nn.Module):
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
    conv = Conv1dGeneral(1,
                         16,
                         3)
    conv.reset()
    ts = tr.jit.script(conv)
    conv.prepare_for_inference()
    ts = tr.jit.script(conv)
    conv.set_cached(True)
    ts = tr.jit.script(conv)
    conv.set_cached(False)
    ts = tr.jit.script(conv)
    exit()


    out_channels = [8] * 4
    tcn = TCN(out_channels, cond_dim=3, padding=0, is_causal=True, is_cached=True)
    log.info(tcn.calc_receptive_field())
    audio = tr.rand((1, 1, 65536))
    cond = tr.rand((1, 3))
    # cond = None
    out = tcn.forward(audio, cond)
    log.info(out.shape)
