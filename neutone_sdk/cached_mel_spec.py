import logging
import os

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
        n_mels: int = 128,
        normalized: bool = False,
        center: bool = True,
        use_debug_mode: bool = True,
    ) -> None:
        assert center, "center must be True, causal mode is not supported yet"
        super().__init__()
        if use_debug_mode:
            assert n_fft % 2 == 0, "n_fft must be even"
            assert (n_fft // 2) % hop_len == 0, "n_fft // 2 must be divisible by hop_len"
            assert hop_len < n_fft, "hop_len must be less than n_fft"
        self.n_ch = n_ch
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.use_debug_mode = use_debug_mode
        self.mel_spec = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_len,
            normalized=normalized,
            n_mels=n_mels,
            center=False,
        )
        self.padding = n_fft // 2
        self.padding_frames = self.padding // hop_len
        self.cache_ahead = CircularInplaceTensorQueue(
            n_ch, self.padding, use_debug_mode
        )
        self.cache_behind = CircularInplaceTensorQueue(
            n_ch, self.padding, use_debug_mode
        )
        self.register_buffer("padding_ahead", tr.zeros((n_ch, self.padding)))
        self.register_buffer("padding_behind", tr.zeros((n_ch, self.padding)))
        self.cache_ahead.push(self.padding_ahead)
        self.cache_behind.push(self.padding_behind)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2, "input audio must have shape (n_ch, n_samples)"
            assert x.size(0) == self.n_ch, "input audio n_ch is incorrect"
            assert (
                x.size(1) % self.hop_len == 0
            ), "input audio n_samples must be divisible by hop_len"
        n_samples = x.size(1)
        n_frames = n_samples // self.hop_len + 1
        lookahead_idx = min(n_samples, self.padding)
        x = tr.cat([self.padding_ahead, x, self.padding_behind], dim=1)
        # if lookahead_idx < n_samples:
        #     x = tr.cat([self.padding_ahead, x, self.padding_behind], dim=1)
        # else:
        #     x = tr.cat([self.padding_ahead, self.padding_behind], dim=1)
        log.info(f"x = {x}")
        spec = self.mel_spec(x)
        spec = spec[:, :, :n_frames]
        self.cache_ahead.push(x[:, :lookahead_idx])
        self.cache_ahead.fill(self.padding_ahead)
        self.cache_behind.fill(self.padding_behind)
        self.cache_behind.push(x[:, -lookahead_idx:])
        return spec

    @tr.jit.export
    def get_delay_samples(self) -> int:
        return self.padding

    @tr.jit.export
    def reset(self) -> None:
        self.cache_ahead.reset()
        self.cache_behind.reset()
        self.padding_left.zero_()
        self.padding_right.zero_()
        self.cache_ahead.push(self.padding_ahead)
        self.cache_behind.push(self.padding_behind)


def test_cached_mel_spec():
    sr = 44100
    n_ch = 1
    n_fft = 4
    hop_len = 1
    n_mels = 1
    total_n_samples = 7 * hop_len

    audio = tr.rand(n_ch, total_n_samples)
    log.info(f"audio = {audio}")
    mel_spec = MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_len,
        n_mels=n_mels,
        center=True,
        pad_mode="constant",
    )
    cached_mel_spec = CachedMelSpec(sr, n_ch, n_fft, hop_len, n_mels)

    spec = mel_spec(audio)
    cached_spec = cached_mel_spec(audio)
    assert tr.allclose(spec, cached_spec)

    chunks = []
    min_chunk_size = hop_len
    max_chunk_size = 1 * hop_len
    curr_idx = 0
    while curr_idx < total_n_samples - max_chunk_size:
        chunk_size = tr.randint(min_chunk_size, max_chunk_size + 1, (1,)).item()
        chunks.append(audio[:, curr_idx:curr_idx + chunk_size])
        curr_idx += chunk_size
    if curr_idx < total_n_samples:
        chunks.append(audio[:, curr_idx:])

    spec_chunks = []
    for chunk in chunks:
        spec_chunk = cached_mel_spec(chunk)
        spec_chunks.append(spec_chunk)
    chunked_spec = tr.cat(spec_chunks, dim=2)
    log.info(f"        spec = {spec}")
    log.info(f"chunked_spec = {chunked_spec}")
    assert tr.allclose(spec, chunked_spec)


if __name__ == "__main__":
    test_cached_mel_spec()
