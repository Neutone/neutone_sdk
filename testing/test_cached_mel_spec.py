import logging
import os

import torch as tr
from torchaudio.transforms import MelSpectrogram

from neutone_sdk.cached_mel_spec import CachedMelSpec

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def test_cached_mel_spec():
    # Setup
    tr.set_printoptions(precision=1)
    tr.random.manual_seed(42)

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
    cached_mel_spec = CachedMelSpec(
        sr=sr, n_ch=n_ch, n_fft=n_fft, hop_len=hop_len, n_mels=n_mels
    )

    # Test delay
    delay_samples = cached_mel_spec.get_delay_samples()
    assert delay_samples == n_fft // 2 - hop_len

    # Test processing all audio at once
    spec = mel_spec(audio)
    delay_frames = cached_mel_spec.get_delay_frames()
    cached_spec = cached_mel_spec(audio)
    cached_spec = cached_spec[:, :, delay_frames:]
    # log.info(f"       spec = {spec}")
    # log.info(f"cached_spec = {cached_spec}")
    assert tr.allclose(spec[:, :, : cached_spec.size(2)], cached_spec)
    cached_mel_spec.reset()

    # Test processing audio in chunks (random chunk size)
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
    chunks.append(
        tr.zeros(n_ch, cached_mel_spec.get_delay_samples() + cached_mel_spec.hop_len)
    )

    spec_chunks = []
    for chunk in chunks:
        spec_chunk = cached_mel_spec(chunk)
        spec_chunks.append(spec_chunk)
    chunked_spec = tr.cat(spec_chunks, dim=2)
    chunked_spec = chunked_spec[:, :, delay_frames:]
    # log.info(f"        spec = {spec}")
    # log.info(f"chunked_spec = {chunked_spec}")
    assert tr.allclose(spec, chunked_spec)
    log.info("test_cached_mel_spec passed!")


if __name__ == "__main__":
    test_cached_mel_spec()
