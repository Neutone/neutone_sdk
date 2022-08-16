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
    def __init__(self, in_sr: int, model_sr: int) -> None:
        super().__init__()
        self.should_resample = None
        self.in_sr = None
        self.model_sr = None
        self.set_sample_rates(in_sr, model_sr)

    def set_sample_rates(self, in_sr: int, model_sr: int) -> None:
        self.should_resample = in_sr != model_sr
        self.in_sr = in_sr
        self.model_sr = model_sr

    @abstractmethod
    def process_in(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def process_out(self, x: Tensor, out_n: int) -> Tensor:
        pass


# TODO(cm): make this torchscript compatible
class PTResampler(nn.Module):
    def __init__(self, in_sr: int, model_sr: int, align_corners: bool = True) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.in_resampler = None
        self.out_resampler = None
        if in_sr != model_sr:
            self.in_resampler = Resample(orig_freq=in_sr, new_freq=model_sr)
            self.out_resampler = Resample(orig_freq=model_sr, new_freq=in_sr)

    def process_in(self, x: Tensor) -> Tensor:
        if self.in_resampler is not None:
            x = self.in_resampler(x)
        return x

    def process_out(self, x: Tensor, out_n: int) -> Tensor:
        if self.out_resampler is not None:
            corner_1_value = x[:, 0]
            corner_2_value = x[:, -1]
            x = self.out_resampler(x)
            if x.shape[1] > out_n:
                x = x[:, :out_n]
            if self.align_corners:
                x[:, 0] = corner_1_value
                x[:, -1] = corner_2_value
        return x


class InterpolationResampler(ResampleSandwich):
    def calc_out_n(self, in_n: int, in_sr: int, out_sr: int) -> int:
        out_n = in_n * out_sr / in_sr
        out_n = math.ceil(out_n)
        return out_n

    def interpolate(self, x: Tensor, out_n: int) -> Tensor:
        # TODO(christhetree): this allocates memory
        x = x.unsqueeze(0)
        x = F.interpolate(x,
                          out_n,
                          mode='linear',
                          align_corners=True)
        x = x.squeeze(0)
        return x

    def process_in(self, x: Tensor) -> Tensor:
        if self.should_resample:
            in_n = x.shape[1]
            out_n = self.calc_out_n(in_n, self.in_sr, self.model_sr)
            x = self.interpolate(x, out_n)
        return x

    def process_out(self, x: Tensor, out_n: int) -> Tensor:
        if self.should_resample:
            x = self.interpolate(x, out_n)
        return x
