import logging
import os
from typing import Optional, Callable

import torch as tr
from torch import Tensor
from torch import nn
from torchaudio.transforms import MelSpectrogram

from neutone_sdk import CircularInplaceTensorQueue

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class CachedMelSpec(nn.Module):
    def __init__(
        self,
        sr: int,
        n_ch: int,
        n_fft: int = 2048,
        hop_len: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_mels: int = 128,
        window_fn: Callable[..., Tensor] = tr.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        use_debug_mode: bool = True,
    ) -> None:
        """
        Creates a Mel spectrogram that supports streaming of a centered, non-causal
        Mel spectrogram operation that uses zero padding. Using this will result in
        audio being delayed by (n_fft / 2) - hop_len samples. When calling forward,
        the input audio block length must be a multiple of the hop length.

        Parameters:
            sr (int): Sample rate of the audio
            n_ch (int): Number of audio channels
            n_fft (int): STFT n_fft (must be even)
            hop_len (int): STFT hop length (must divide into n_fft // 2)
            f_min (float): Minimum frequency of the Mel filterbank
            f_max (float): Maximum frequency of the Mel filterbank
            n_mels (int): Number of mel filterbank bins
            window_fn (Callable[..., Tensor]): A function to create a window tensor
            power (float): Exponent for the magnitude spectrogram (must be > 0)
            normalized (bool): Whether to normalize the mel spectrogram or not
            center (bool): Whether to center the mel spectrogram (must be True)
            use_debug_mode (bool): Whether to use debug mode or not
        """
        super().__init__()
        assert center, "center must be True, causal mode is not supported yet"
        assert n_fft % 2 == 0, "n_fft must be even"
        assert (n_fft // 2) % hop_len == 0, "n_fft // 2 must be divisible by hop_len"
        self.n_ch = n_ch
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.use_debug_mode = use_debug_mode
        self.mel_spec = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            center=False,  # We use a causal STFT since we do the padding ourselves
        )
        self.padding_n_samples = self.n_fft - self.hop_len
        self.cache = CircularInplaceTensorQueue(
            n_ch, self.padding_n_samples, use_debug_mode
        )
        self.register_buffer("padding", tr.zeros((n_ch, self.padding_n_samples)))
        self.cache.push(self.padding)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the Mel spectrogram of the input audio tensor. Supports streaming as
        long as the input audio tensor is a multiple of the hop length.
        """
        if self.use_debug_mode:
            assert x.ndim == 2, "input audio must have shape (n_ch, n_samples)"
            assert x.size(0) == self.n_ch, "input audio n_ch is incorrect"
            assert (
                x.size(1) % self.hop_len == 0
            ), "input audio n_samples must be divisible by hop_len"
        # Compute the Mel spec
        n_samples = x.size(1)
        n_frames = n_samples // self.hop_len
        padded_x = tr.cat([self.padding, x], dim=1)
        padded_spec = self.mel_spec(padded_x)
        spec = padded_spec[:, :, -n_frames:]

        # Update the cache and padding
        padding_idx = min(n_samples, self.padding_n_samples)
        self.cache.push(x[:, -padding_idx:])
        self.cache.fill(self.padding)
        return spec

    def prepare_for_inference(self) -> None:
        """
        Prepares the cached Mel spectrogram for inference by disabling debug mode.
        """
        self.cache.use_debug_mode = False
        self.use_debug_mode = False

    @tr.jit.export
    def get_delay_samples(self) -> int:
        """
        Returns the number of samples of delay of the cached Mel spectrogram.
        """
        return (self.n_fft // 2) - self.hop_len

    @tr.jit.export
    def get_delay_frames(self) -> int:
        """
        Returns the number of frames of delay of the cached Mel spectrogram.
        """
        return self.get_delay_samples() // self.hop_len

    @tr.jit.export
    def reset(self) -> None:
        """
        Resets the cache and padding of the cached Mel spectrogram.
        """
        self.cache.reset()
        self.padding.zero_()
        self.cache.push(self.padding)
