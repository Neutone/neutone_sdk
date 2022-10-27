import logging
import math
import os
from abc import abstractmethod, ABC

import torch as tr
import torch.nn.functional as F
from torch import Tensor, nn
from torchaudio.transforms import Resample

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ChannelNormalizerSandwich(nn.Module):
    """
    Converts between mono and stereo channels as required without allocating memory.
    """
    def __init__(self, use_debug_mode: bool = True) -> None:
        super().__init__()
        self.use_debug_mode = use_debug_mode
        self.half_scalar = tr.tensor(0.5)

    def forward(self,
                x: Tensor,
                should_be_mono: bool,
                out_buffer: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(0) <= 2
            assert out_buffer.ndim == 2
            assert out_buffer.size(0) == 2
            assert out_buffer.size(1) >= x.size(1)
        n_ch = x.size(0)
        n_samples = x.size(1)
        if should_be_mono and n_ch == 2:
            out = out_buffer[0:1, 0:n_samples]
            tr.add(x[0:1, :], x[1:2, :], out=out)
            tr.mul(out, self.half_scalar, out=out)
            x = out
        elif not should_be_mono and n_ch == 1:
            out_buffer[0:1, 0:n_samples] = x
            out_buffer[1:2, 0:n_samples] = x
            x = out_buffer
        return x


class ResampleSandwich(ABC, nn.Module):
    def __init__(self, in_sr: int, out_sr: int, in_bs: int, use_debug_mode: bool = True) -> None:
        """
        Common interface for resampling sandwiches.

        Args:
            in_sr: incoming sampling rate
            out_sr: desired sampling rate
            in_bs: incoming buffer size
            use_debug_mode: enable debug mode
        """
        super().__init__()
        self.use_debug_mode = use_debug_mode

        if self.use_debug_mode:
            assert in_sr > 0
            assert out_sr > 0
            assert in_bs >= 2

        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs

        self.out_bs = None
        self.set_sample_rates(in_sr, out_sr, in_bs)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        """
        Set the sampling rates of the sandwich. This should be called every time in_sr, out_sr, or in_bs changes.
        """
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs >= 2

    def is_resampling(self) -> bool:
        return self.in_bs != self.out_bs

    @abstractmethod
    def process_in(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def process_out(self, x: Tensor) -> Tensor:
        pass


# TODO(cm): make this torchscript compatible
class PTResampler(ResampleSandwich):
    """
    Antialiasing resampling using the default PyTorch audio resampling implementation.
    Slower, dynamically allocates memory, and is not TorchScript compatible.
    """
    def __init__(self,
                 in_sr: int,
                 out_sr: int,
                 in_bs: int,
                 resampling_method: str = "sinc_interpolation",
                 align_corners: bool = True,
                 use_debug_mode: bool = True) -> None:
        self.resampling_method = resampling_method
        self.align_corners = align_corners
        self.in_resampler = None
        self.out_resampler = None
        super().__init__(in_sr, out_sr, in_bs, use_debug_mode)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs

        self.in_resampler = Resample(orig_freq=self.in_sr,
                                     new_freq=self.out_sr,
                                     resampling_method=self.resampling_method)
        self.out_resampler = Resample(orig_freq=self.out_sr,
                                      new_freq=self.in_sr,
                                      resampling_method=self.resampling_method)

        tmp = self.in_resampler(tr.zeros((2, self.in_bs)))
        self.out_bs = tmp.size(1)
        if self.use_debug_mode:
            assert self.out_bs >= 2

    def process_in(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == self.in_bs
        if self.is_resampling():
            corner_1_value = x[:, 0]
            corner_2_value = x[:, -1]
            x = self.in_resampler(x)
            if self.align_corners:
                x[:, 0] = corner_1_value
                x[:, -1] = corner_2_value
        return x

    def process_out(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == self.out_bs
        if self.is_resampling():
            corner_1_value = x[:, 0]
            corner_2_value = x[:, -1]
            x = self.out_resampler(x)
            if x.size(1) > self.in_bs:
                x = x[:, :self.in_bs]
            if self.align_corners:
                x[:, 0] = corner_1_value
                x[:, -1] = corner_2_value
        return x


class InterpolationResampler(ResampleSandwich):
    """
    Interpolation-based resampling using the default PyTorch linear interpolation implementation.
    Dynamically allocates memory.
    """
    def _process(self, x: Tensor, in_bs: int, out_bs: int) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2
            assert x.size(1) == in_bs
        if self.is_resampling():
            x = x.unsqueeze(0)
            x = F.interpolate(x,
                              out_bs,
                              mode='linear',
                              align_corners=True)
            x = x.squeeze(0)
        return x

    def process_in(self, x: Tensor) -> Tensor:
        return self._process(x, self.in_bs, self.out_bs)

    def process_out(self, x: Tensor) -> Tensor:
        return self._process(x, self.out_bs, self.in_bs)


class InplaceInterpolationResampler(ResampleSandwich):
    """
    Interpolation-based resampling using a custom implementation.
    Does not dynamically allocate memory and is ~40% faster than the PyTorch implementation for common sampling rates.
    """
    def __init__(self,
                 in_n_ch: int,
                 out_n_ch: int,
                 in_sr: int,
                 out_sr: int,
                 in_bs: int,
                 use_debug_mode: bool = True) -> None:
        self.in_n_ch = in_n_ch
        self.out_n_ch = out_n_ch

        # Buffers required for process_in
        self.interp_in = None
        self.floor_in = None
        self.ceil_in = None
        self.in_a = None
        self.in_b = None
        # Buffers required for process_out
        self.interp_out = None
        self.floor_out = None
        self.ceil_out = None
        self.out_a = None
        self.out_b = None

        super().__init__(in_sr, out_sr, in_bs, use_debug_mode)

    def set_sample_rates(self, in_sr: int, out_sr: int, in_bs: int) -> None:
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.in_bs = in_bs
        self.out_bs = math.ceil(self.in_bs * self.out_sr / self.in_sr)
        if self.use_debug_mode:
            assert self.out_bs >= 2

        scaling_factor_in = (self.in_bs - 1) / (self.out_bs - 1)
        interp_in = tr.tensor([float(_) for _ in range(self.out_bs)]) * scaling_factor_in
        floor_in = tr.floor(interp_in).to(tr.long)
        self.floor_in = tr.clip(floor_in, 0, self.in_bs - 1)  # Prevents floating point errors
        ceil_in = tr.ceil(interp_in).to(tr.long)
        self.ceil_in = tr.clip(ceil_in, 0, self.in_bs - 1)  # Prevents floating point errors
        self.interp_in = tr.clip(interp_in - self.floor_in, 0.0, 1.0)  # Prevents floating point errors
        self.in_a = tr.zeros((self.in_n_ch, self.out_bs))
        self.in_b = tr.zeros((self.in_n_ch, self.out_bs))

        scaling_factor_out = (self.out_bs - 1) / (self.in_bs - 1)
        interp_out = tr.tensor([float(_) for _ in range(self.in_bs)]) * scaling_factor_out
        floor_out = tr.floor(interp_out).to(tr.long)
        self.floor_out = tr.clip(floor_out, 0, self.out_bs - 1)  # Prevents floating point errors
        ceil_out = tr.ceil(interp_out).to(tr.long)
        self.ceil_out = tr.clip(ceil_out, 0, self.out_bs - 1)  # Prevents floating point errors
        self.interp_out = tr.clip(interp_out - self.floor_out, 0.0, 1.0)  # Prevents floating point errors
        self.out_a = tr.zeros((self.out_n_ch, self.in_bs))
        self.out_b = tr.zeros((self.out_n_ch, self.in_bs))

    def _process(self,
                 x: Tensor,
                 n_ch: int,
                 in_bs: int,
                 interp_t: Tensor,
                 floor_t: Tensor,
                 ceil_t: Tensor,
                 a_t: Tensor,
                 b_t: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.shape == (n_ch, in_bs)
        if not self.is_resampling():
            return x
        tr.index_select(x, 1, floor_t, out=a_t)
        tr.index_select(x, 1, ceil_t, out=b_t)
        tr.sub(b_t, a_t, out=b_t)
        tr.mul(b_t, interp_t, out=b_t)
        tr.add(a_t, b_t, out=a_t)
        return a_t

    def process_in(self, x: Tensor) -> Tensor:
        return self._process(x,
                             self.in_n_ch,
                             self.in_bs,
                             self.interp_in,
                             self.floor_in,
                             self.ceil_in,
                             self.in_a,
                             self.in_b)

    def process_out(self, x: Tensor) -> Tensor:
        return self._process(x,
                             self.out_n_ch,
                             self.out_bs,
                             self.interp_out,
                             self.floor_out,
                             self.ceil_out,
                             self.out_a,
                             self.out_b)
