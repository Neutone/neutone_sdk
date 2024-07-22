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
            assert (
                n_fft // 2
            ) % hop_len == 0, "n_fft // 2 must be divisible by hop_len"
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
        self.queue = CircularInplaceTensorQueue(n_ch, self.n_fft, use_debug_mode)
        self.register_buffer("padding", tr.zeros((n_ch, self.n_fft)))
        self.queue.push(self.padding)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert x.ndim == 2, "input audio must have shape (n_ch, n_samples)"
            assert x.size(0) == self.n_ch, "input audio n_ch is incorrect"
            assert (
                x.size(1) % self.hop_len == 0
            ), "input audio n_samples must be divisible by hop_len"
        n_samples = x.size(1)
        n_frames = n_samples // self.hop_len
        padded_x = tr.cat([self.padding, x], dim=1)
        # log.info(f"padded_x = {padded_x}")
        spec = self.mel_spec(padded_x)
        spec = spec[:, :, :n_frames]

        lookahead_idx = min(n_samples, self.n_fft)
        self.queue.push(x[:, -lookahead_idx:])
        self.queue.fill(self.padding)
        return spec

    @tr.jit.export
    def get_delay_samples(self) -> int:
        return self.n_fft // 2

    @tr.jit.export
    def get_delay_frames(self) -> int:
        return self.get_delay_samples() // self.hop_len

    @tr.jit.export
    def reset(self) -> None:
        self.queue.reset()
        self.padding.zero_()
        self.queue.push(self.padding)


def test_cached_mel_spec():
    tr.set_printoptions(precision=2)
    tr.random.manual_seed(0)

    sr = 44100
    n_ch = 1
    n_fft = 2048
    hop_len = 128
    n_mels = 16
    total_n_samples = 1000 * hop_len

    audio = tr.rand(n_ch, total_n_samples)
    # log.info(f"audio = {audio}")
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
    delay_frames = cached_mel_spec.get_delay_frames()
    cached_spec = cached_mel_spec(audio)
    cached_spec = cached_spec[:, :, delay_frames:]
    # log.info(f"       spec = {spec}")
    # log.info(f"cached_spec = {cached_spec}")
    assert tr.allclose(spec[:, :, : cached_spec.size(2)], cached_spec)
    cached_mel_spec.reset()

    chunks = []
    min_chunk_size = 1
    max_chunk_size = 100
    curr_idx = 0
    while curr_idx < total_n_samples - max_chunk_size:
        chunk_size = (
            tr.randint(min_chunk_size, max_chunk_size + 1, (1,)).item() * hop_len
        )
        chunks.append(audio[:, curr_idx : curr_idx + chunk_size])
        curr_idx += chunk_size
    if curr_idx < total_n_samples:
        chunks.append(audio[:, curr_idx:])
    chunks.append(tr.zeros(n_ch, cached_mel_spec.n_fft))

    spec_chunks = []
    for chunk in chunks:
        spec_chunk = cached_mel_spec(chunk)
        spec_chunks.append(spec_chunk)
    chunked_spec = tr.cat(spec_chunks, dim=2)
    chunked_spec = chunked_spec[:, :, delay_frames:]
    # log.info(f"        spec = {spec}")
    # log.info(f"chunked_spec = {chunked_spec}")
    assert tr.allclose(spec, chunked_spec[:, :, : spec.size(2)])


if __name__ == "__main__":
    test_cached_mel_spec()
