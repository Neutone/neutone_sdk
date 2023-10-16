import logging
import os
import random

import torch as tr
import torch.nn.functional as F
from tqdm import tqdm

from neutone_sdk.sandwich import LinearResampler, InplaceLinearResampler, Inplace4pHermiteResampler

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def test_linear_resamplers(n_trials: int = 1000, in_n_ch: int = 2, out_n_ch: int = 2) -> None:
    random.seed(42)
    tr.manual_seed(42)
    sampling_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000]
    buffer_sizes = [64, 128, 256, 512, 1024, 2048]

    resampler = LinearResampler(48000, 48000, 512)
    inplace_resampler = InplaceLinearResampler(
        in_n_ch, out_n_ch, 48000, 48000, 512
    )

    for _ in tqdm(range(n_trials)):
        sr_a = random.choice(sampling_rates)
        sr_b = random.choice(sampling_rates)
        in_bs = random.choice(buffer_sizes)

        resampler.set_sample_rates(sr_a, sr_b, in_bs)
        inplace_resampler.set_sample_rates(sr_a, sr_b, in_bs)
        # Check inplace linear resampler internal values are correct for matching the ends exactly
        assert inplace_resampler.x_in[0] == 0.0
        assert inplace_resampler.x_in[-1] == 0.0 or inplace_resampler.x_in[-1] == 1.0

        in_audio = tr.rand((in_n_ch, in_bs))
        in_linear = resampler.process_in(in_audio)
        in_linear_inplace = inplace_resampler.process_in(in_audio)
        out_bs = in_linear.size(1)
        assert in_linear.shape == in_linear_inplace.shape

        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(in_linear[:, 1:-1], in_linear_inplace[:, 1:-1], atol=1e-6)
        assert tr.allclose(in_linear[:, [0, -1]], in_linear_inplace[:, [0, -1]], atol=1e-3)
        in_interpolated = F.interpolate(
            in_audio.unsqueeze(0), out_bs, mode="linear", align_corners=True
        ).squeeze(0)
        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(in_linear_inplace[:, 1:-1], in_interpolated[:, 1:-1], atol=1e-6)
        assert tr.allclose(in_linear_inplace[:, [0, -1]], in_interpolated[:, [0, -1]], atol=1e-3)
        # Check that the ends match exactly
        assert tr.equal(in_linear_inplace[:, [0, -1]], in_audio[:, [0, -1]])

        out_audio = tr.rand((out_n_ch, out_bs))
        out_linear = resampler.process_out(out_audio)
        out_linear_inplace = inplace_resampler.process_out(out_audio)
        assert out_linear.shape == out_linear_inplace.shape
        assert out_linear.size(1) == in_bs

        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(out_linear[:, 1:-1], out_linear_inplace[:, 1:-1], atol=1e-6)
        assert tr.allclose(out_linear[:, [0, -1]], out_linear_inplace[:, [0, -1]], atol=1e-3)
        out_interpolated = F.interpolate(
            out_audio.unsqueeze(0), in_bs, mode="linear", align_corners=True
        ).squeeze(0)
        # PyTorch interpolation does not match the ends exactly, hence two asserts
        assert tr.allclose(out_linear_inplace[:, 1:-1], out_interpolated[:, 1:-1], atol=1e-6)
        assert tr.allclose(out_linear_inplace[:, [0, -1]], out_interpolated[:, [0, -1]], atol=1e-3)
        # Check that the ends match exactly
        assert tr.equal(out_linear_inplace[:, [0, -1]], out_audio[:, [0, -1]])


test_linear_resamplers()
