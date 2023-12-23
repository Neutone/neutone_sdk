import itertools
import logging
import os
import random
from typing import Union, Tuple

import torch as tr
from torch import nn
from tqdm import tqdm

from neutone_sdk.conv import Conv1dGeneral

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _test_against_conv_torch(in_channels: int,
                             out_channels: int,
                             kernel_size: int,
                             padding: Union[str, int, Tuple[int]],
                             dilation: int,
                             causal: bool,
                             padding_mode: str = "zeros",
                             batch_size: int = 1,
                             block_size: int = 128,
                             n_blocks: int = 32) -> None:
    conv_gen = Conv1dGeneral(in_channels,
                             out_channels,
                             kernel_size,
                             padding=padding,
                             dilation=dilation,
                             padding_mode=padding_mode,
                             causal=causal,
                             cached=False,
                             bias=False)
    padding_torch = padding
    if causal and padding == "same":
        assert conv_gen.right_padding == 0
        padding_torch = conv_gen.padded_kernel_size
    conv_torch = nn.Conv1d(in_channels,
                           out_channels,
                           kernel_size,
                           padding=padding_torch,
                           dilation=dilation,
                           padding_mode=padding_mode,
                           bias=False)

    conv_torch.weight = nn.Parameter(conv_gen.conv1d.weight.clone())
    if conv_torch.bias is not None:
        conv_torch.bias = nn.Parameter(conv_gen.conv1d.bias.clone())

    audio = tr.rand((batch_size, in_channels, n_blocks * block_size))
    out_torch = conv_torch(audio)
    out_gen = conv_gen(audio)
    # Regular torch conv1d can't do causal convs so get rid of the extra right samples
    if causal and padding != "valid":
        if conv_gen.left_padding > 0:
            out_torch = out_torch[..., :-conv_gen.left_padding]
    assert out_gen.shape == out_torch.shape
    assert tr.allclose(out_gen, out_torch)

    conv_gen.set_cached(True)
    out_blocks = []
    for idx in range(n_blocks):
        audio_block = audio[..., idx * block_size:(idx + 1) * block_size]
        out_blocks.append(conv_gen(audio_block))
    out_cached = tr.cat(out_blocks, dim=-1)

    delay_samples = conv_gen.calc_delay_samples()
    if delay_samples > 0:
        out_cached = out_cached[..., delay_samples:]
        out_torch = out_torch[..., :-delay_samples]
    if out_cached.size(-1) > out_torch.size(-1):
        out_cached = Conv1dGeneral.causal_crop(out_cached, out_torch.size(-1))
    else:
        out_torch = Conv1dGeneral.causal_crop(out_torch, out_cached.size(-1))
    assert out_cached.shape == out_torch.shape
    assert tr.allclose(out_cached, out_torch)


def test_conv1d_general():
    causal_flags = [False, True]
    in_channels = [1, 2]
    out_ch = 1
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    dilations = [1, 2, 3, 4, 5, 6, 7, 8]
    max_rand_padding = 32

    for causal, in_ch, kernel_size, dil in tqdm(itertools.product(causal_flags,
                                                                  in_channels,
                                                                  kernel_sizes,
                                                                  dilations)):
        rand_padding = random.randint(1, max_rand_padding)
        log.info(f"Testing causal={causal}, in_ch={in_ch}, kernel_size={kernel_size}, dil={dil}, rand_padding={rand_padding}")
        _test_against_conv_torch(in_ch, out_ch, kernel_size, padding="same", dilation=dil, causal=causal)
        _test_against_conv_torch(in_ch, out_ch, kernel_size, padding="valid", dilation=dil, causal=causal)
        _test_against_conv_torch(in_ch, out_ch, kernel_size, padding=0, dilation=dil, causal=causal)
        _test_against_conv_torch(in_ch, out_ch, kernel_size, padding=rand_padding, dilation=dil, causal=causal)


if __name__ == "__main__":
    test_conv1d_general()
