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


def test_dynamic_bs() -> None:
    conv_gen = Conv1dGeneral(in_channels=2,
                             out_channels=16,
                             kernel_size=5,
                             padding="same",
                             dilation=2,
                             causal=False,
                             cached=True,
                             use_dynamic_bs=True)
    for bs in range(64):
        audio = tr.rand((bs, 2, 128))
        out = conv_gen(audio)
        assert out.shape == (bs, 16, 128)


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
                             cached=False)
    padding_torch = padding
    if causal and padding == "same":
        # torch.nn.Conv1d doesn't support causal convs so we need to add the causal
        # padding to both sides and then remove it from the right side later
        assert conv_gen.padding_r == 0
        padding_torch = conv_gen.padding_l
    conv_torch = nn.Conv1d(in_channels,
                           out_channels,
                           kernel_size,
                           padding=padding_torch,
                           dilation=dilation,
                           padding_mode=padding_mode)

    # Copy weights and biases for testing
    conv_torch.weight = nn.Parameter(conv_gen.conv1d.weight.clone())
    if conv_torch.bias is not None:
        conv_torch.bias = nn.Parameter(conv_gen.conv1d.bias.clone())

    audio = tr.rand((batch_size, in_channels, n_blocks * block_size))
    out_torch = conv_torch(audio)
    out_gen = conv_gen(audio)
    # torch.nn.Conv1d doesn't support causal convs so get rid of the extra right samples
    if causal and padding != "valid":
        if conv_gen.padding_l > 0:
            out_torch = out_torch[..., :-conv_gen.padding_l]
    assert out_gen.shape == out_torch.shape
    assert tr.allclose(out_gen, out_torch)

    conv_gen.set_cached(True)
    out_blocks = []
    for idx in range(n_blocks):
        audio_block = audio[..., idx * block_size:(idx + 1) * block_size]
        out_block = conv_gen(audio_block)
        out_blocks.append(out_block)
    assert all(b.size(-1) == block_size for b in out_blocks)
    out_cached = tr.cat(out_blocks, dim=-1)

    delay_samples = conv_gen.get_delay_samples()
    if delay_samples > 0:
        # Remove the delay samples from the beginning of the cached output to align
        # it with not cached output
        out_cached = out_cached[..., delay_samples:]
        # Remove the delay samples from the end of the not cached output since they were
        # never computed by the cached convolution
        out_torch = out_torch[..., :-delay_samples]
    # Different padding modes can result in different output lengths of out_torch,
    # so we need to crop the longer one to align it with the shorter one
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
        rand_pad = random.randint(1, max_rand_padding)
        log.info(f"Testing causal={causal}, "
                 f"in_ch={in_ch}, "
                 f"kernel_size={kernel_size}, "
                 f"dil={dil}, "
                 f"rand_pad={rand_pad}")
        _test_against_conv_torch(
            in_ch, out_ch, kernel_size, padding="same", dilation=dil, causal=causal)
        _test_against_conv_torch(
            in_ch, out_ch, kernel_size, padding="valid", dilation=dil, causal=causal)
        _test_against_conv_torch(
            in_ch, out_ch, kernel_size, padding=0, dilation=dil, causal=causal)
        _test_against_conv_torch(
            in_ch, out_ch, kernel_size, padding=rand_pad, dilation=dil, causal=causal)


def _test_get_delay_samples(in_channels: int,
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
                             bias=False,
                             causal=causal,
                             cached=False)
    # Create a dirac delta kernel such that the output is the same as the input
    conv_gen.conv1d.weight.data.fill_(0.0)
    if not causal and padding == "same":
        if kernel_size % 2 == 0:
            conv_gen.conv1d.weight.data[..., kernel_size // 2 - 1] = 1.0
        else:
            conv_gen.conv1d.weight.data[..., kernel_size // 2] = 1.0
    else:
        conv_gen.conv1d.weight.data[..., -1] = 1.0

    audio = tr.rand((batch_size, in_channels, n_blocks * block_size))
    out = conv_gen(audio)

    # Always use causal crop since padding is "same" for the non-causal case
    if out.size(-1) > audio.size(-1):
        out = Conv1dGeneral.causal_crop(out, audio.size(-1))
    elif out.size(-1) < audio.size(-1):
        audio = Conv1dGeneral.causal_crop(audio, out.size(-1))

    # assert out.shape == audio.shape
    # assert tr.allclose(out, audio)

    # Audio should have no zero values for measuring delay
    audio = tr.rand((batch_size, in_channels, n_blocks * block_size)) + 0.01
    conv_gen.set_cached(True)
    out_blocks = []
    for idx in range(n_blocks):
        audio_block = audio[..., idx * block_size:(idx + 1) * block_size]
        out_block = conv_gen(audio_block)
        out_blocks.append(out_block)
    assert all(b.size(-1) == block_size for b in out_blocks)
    out_cached = tr.cat(out_blocks, dim=-1)

    delay_samples = conv_gen.get_delay_samples()
    # Find the number of zeros which represents the delay
    n_zeros = tr.sum(out_cached == 0.0, dim=-1).squeeze().item()

    assert n_zeros == delay_samples
    if delay_samples > 0:
        out_cached = out_cached[..., delay_samples:]
        audio = audio[..., :-delay_samples]

    assert out_cached.shape == audio.shape
    assert tr.allclose(out_cached, audio)


def test_get_delay_samples() -> None:
    causal_flags = [False, True]
    kernel_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    # kernel_sizes = [2, 3, 4, 5, 6, 7, 8]
    dilations = [1, 2, 3, 4, 5, 6, 7, 8]
    # dilations = [1]
    in_ch = 1
    max_rand_padding = 32

    for causal, kernel_size, dil in tqdm(itertools.product(causal_flags,
                                                           kernel_sizes,
                                                           dilations)):
        if not causal and kernel_size % 2 == 0 and dil > 1:
            # Creating a direc delta kernel is impossible in this case
            continue
        rand_pad = random.randint(1, max_rand_padding)
        log.info(f"Testing causal={causal}, "
                 f"in_ch={in_ch}, "
                 f"kernel_size={kernel_size}, "
                 f"dil={dil}, "
                 f"rand_pad={rand_pad}")
        _test_get_delay_samples(
            in_ch, in_ch, kernel_size, padding="same", dilation=dil, causal=causal)
        _test_get_delay_samples(
            in_ch, in_ch, kernel_size, padding="valid", dilation=dil, causal=causal)


if __name__ == "__main__":
    test_dynamic_bs()
    test_conv1d_general()
    test_get_delay_samples()
