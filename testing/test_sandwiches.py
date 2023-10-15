import logging
import os
import random

import torch as tr
import torch.nn.functional as F
from tqdm import tqdm

from neutone_sdk.sandwich import InterpolationResampler, InplaceLinearResampler

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def test_interpolation_resamplers(in_n_ch: int = 2, out_n_ch: int = 2) -> None:
    random.seed(42)
    tr.manual_seed(42)
    sampling_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000]
    buffer_sizes = [64, 128, 256, 512, 1024, 2048]
    trials = 1000

    resampler = InterpolationResampler(48000, 48000, 512)
    inplace_resampler = InplaceLinearResampler(
        in_n_ch, out_n_ch, 48000, 48000, 512
    )

    for _ in tqdm(range(trials)):
        sr_a = random.choice(sampling_rates)
        sr_b = random.choice(sampling_rates)
        in_bs = random.choice(buffer_sizes)

        resampler.set_sample_rates(sr_a, sr_b, in_bs)
        inplace_resampler.set_sample_rates(sr_a, sr_b, in_bs)

        in_audio = tr.rand((in_n_ch, in_bs))
        resampled_1 = resampler.process_in(in_audio)
        resampled_2 = inplace_resampler.process_in(in_audio)
        out_bs = resampled_1.size(1)
        assert resampled_1.shape == resampled_2.shape
        assert tr.allclose(resampled_1, resampled_2, atol=1e-5)
        resampled_3 = F.interpolate(
            in_audio.unsqueeze(0), out_bs, mode="linear", align_corners=True
        ).squeeze(0)
        assert tr.allclose(resampled_1, resampled_3, atol=1e-5)

        out_audio = tr.rand((out_n_ch, out_bs))
        resampled_1 = resampler.process_out(out_audio)
        resampled_2 = inplace_resampler.process_out(out_audio)
        assert resampled_1.shape == resampled_2.shape
        assert resampled_1.size(1) == in_bs
        assert tr.allclose(resampled_1, resampled_2, atol=1e-5)
        resampled_3 = F.interpolate(
            out_audio.unsqueeze(0), in_bs, mode="linear", align_corners=True
        ).squeeze(0)
        assert tr.allclose(resampled_1, resampled_3, atol=1e-5)


test_interpolation_resamplers()
