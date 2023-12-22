import logging
import os
from typing import Optional, Tuple, Union

import torch as tr
import torch.nn.functional as F
from torch import Tensor
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


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
                 padding: Union[str, int, Tuple[int], Tuple[int, int]] = "same",
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

        self.padded_kernel_size = (kernel_size - 1) * dilation
        left_padding, right_padding, left_padding_cached, right_padding_cached = (
            self._calc_padding(kernel_size, stride, padding, dilation, causal))
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.uncached_padding = (left_padding, right_padding)
        self.left_padding_cached = left_padding_cached
        self.right_padding_cached = right_padding_cached

        self.conv1d = nn.Conv1d(in_channels,
                                out_channels,
                                kernel_size=(kernel_size,),
                                stride=(stride,),
                                padding=0,
                                dilation=(dilation,),
                                bias=bias,
                                padding_mode=padding_mode)
        if padding_mode == "zeros":
            # After setting the torch conv padding mode, we need to change it to be
            # compatible with F.pad(). It's strange that this is inconsistent in torch.
            self.padding_mode = "constant"
        self.padding_cached = PaddingCached(in_channels,
                                            left_padding_cached,
                                            use_dynamic_bs,
                                            batch_size,
                                            debug_mode)

    def _calc_padding(self,
                      kernel_size: int,
                      stride: int,
                      padding: Union[str, int, Tuple[int], Tuple[int, int]],
                      dilation: int,
                      causal: bool) -> Tuple[int, int, int, int]:
        if isinstance(padding, tuple) and len(padding) == 1:
            padding = padding[0]
        padded_kernel_size = (kernel_size - 1) * dilation
        if padding == "valid":
            left_padding = 0
            right_padding = 0
        elif padding == "same":
            assert stride == 1, "If padding is 'same', stride must be 1"
            if causal:
                left_padding = padded_kernel_size
                right_padding = 0
            elif padded_kernel_size % 2 == 0:
                left_padding = padded_kernel_size // 2
                right_padding = padded_kernel_size // 2
            else:
                left_padding = padded_kernel_size // 2
                # Favor right padding over left padding just like in nn.Conv1d
                right_padding = padded_kernel_size // 2 + 1
        elif isinstance(padding, int):
            assert padding >= 0
            if causal:
                left_padding = padding
                right_padding = 0
            else:
                left_padding = padding
                right_padding = padding
        else:
            assert len(padding) == 2, "Expected padding to be a tuple of length 2."
            assert padding[0] >= 0 and padding[1] >= 0
            if causal:
                assert padding[1] == 0, "If causal, right padding must be 0"
            left_padding = padding[0]
            right_padding = padding[1]

        left_padding_cached = max(padded_kernel_size, left_padding)
        right_padding_cached = max(padded_kernel_size, right_padding)
        return left_padding, right_padding, left_padding_cached, right_padding_cached

    @tr.jit.export
    def set_cached(self, cached: bool) -> None:
        self.cached = cached
        # Batch size needs to be provided for TorchScript
        self.reset(batch_size=None)

    @tr.jit.export
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
        n_samples = x.size(-1)
        if self.cached:
            x = self.padding_cached(x)
            if self.right_padding > 0:
                # TODO(cm): prevent dynamic memory allocations here
                x = F.pad(x, (0, self.right_padding), mode=self.padding_mode)
        elif self.uncached_padding != (0, 0):
            # TODO(cm): prevent dynamic memory allocations here
            x = F.pad(x, self.uncached_padding, mode=self.padding_mode)
        x = self.conv1d(x)
        if self.cached:
            if self.causal:
                x = self.causal_crop(x, n_samples)
            else:
                x = self.center_crop(x, n_samples)
        return x

    @staticmethod
    def center_crop(x: Tensor, length: int) -> Tensor:
        if x.size(-1) != length:
            assert x.size(-1) > length
            start = (x.size(-1) - length) // 2
            stop = start + length
            x = x[..., start:stop]
        return x

    @staticmethod
    def causal_crop(x: Tensor, length: int) -> Tensor:
        if x.size(-1) != length:
            assert x.size(-1) > length
            x = x[..., -length:]
        return x


if __name__ == "__main__":
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
