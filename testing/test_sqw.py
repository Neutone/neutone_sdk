import itertools
import logging
import math
import os
import random
from typing import Dict, List, Optional

import torch as tr
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from neutone_sdk import WaveformToWaveformBase, SampleQueueWrapper

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TestModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class TestModelWrapper(WaveformToWaveformBase):
    def __init__(self,
                 model: nn.Module = TestModel(),
                 model_sr: int = 48000,
                 model_bs: int = 512,
                 use_debug_mode: bool = True) -> None:
        super().__init__(model, use_debug_mode)
        self.model_sr = model_sr
        self.model_bs = model_bs

    def get_model_name(self) -> str:
        return "test"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Testing."

    def get_model_long_description(self) -> str:
        return "Testing."

    def get_technical_description(self) -> str:
        return "Testing."

    def get_tags(self) -> List[str]:
        return ["test"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [self.model_sr]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [self.model_bs]

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        return self.model.forward(x)


def check_saturation_n(io_bs: int, model_bs: int, saturation_n: int) -> bool:
    size_in = saturation_n
    size_out = 0
    for _ in range(math.lcm(io_bs, model_bs)):
        while size_in >= model_bs:
            size_in -= model_bs
            size_out += model_bs
        if size_out < io_bs:
            return False
        else:
            size_out -= io_bs
        assert size_in >= 0
        assert size_out >= 0
        size_in += io_bs
    return True


def find_saturation_n(io_bs: int, model_bs: int) -> Optional[int]:
    lcm = math.lcm(io_bs, model_bs)
    for n in range(io_bs, lcm + 1, io_bs):
        if check_saturation_n(io_bs, model_bs, n):
            return n
    return None


def check_queue_saturation(io_bs: int, model_bs: int, saturation_n: int) -> bool:
    sr = 48000
    wrapper = TestModelWrapper(model_sr=sr, model_bs=model_bs)
    sqw = SampleQueueWrapper(wrapper, daw_sr=sr, daw_bs=io_bs, model_sr=sr, model_bs=model_bs)
    in_queue = sqw.in_queue
    out_queue = sqw.out_queue

    io_buffer = tr.zeros((2, io_bs))
    model_buffer = tr.zeros((2, model_bs))

    is_saturated = False
    audio_in = tr.rand((2, (io_bs * model_bs) + (2 * saturation_n)))
    blocks_in = tr.split(audio_in, io_bs, dim=1)

    for block_in in blocks_in:
        if block_in.size(1) != io_bs:
            break

        assert in_queue.max_size - in_queue.size >= io_bs
        in_queue.push(block_in)

        if in_queue.size >= saturation_n:
           is_saturated = True

        while in_queue.size >= model_bs:
            in_popped_n = in_queue.pop(model_buffer)
            assert in_popped_n == model_bs
            assert out_queue.max_size - in_queue.size >= model_bs
            out_queue.push(model_buffer)

        if is_saturated:
            out_popped_n = out_queue.pop(io_buffer)
            if out_popped_n != io_bs:
                return False

    return True


def delay_test(wrapper: TestModelWrapper,
               sqw: SampleQueueWrapper,
               daw_sr: int,
               daw_bs: int,
               model_sr: int,
               model_bs: int) -> None:
    wrapper.model_sr = model_sr
    wrapper.model_bs = model_bs
    sqw.set_daw_sample_rate_and_buffer_size(daw_sr, daw_bs)
    expected_delay = sqw.calc_min_delay_samples()
    assert expected_delay >= 0

    n_samples = expected_delay + (2 * max(daw_bs, model_bs))
    audio_in = tr.rand((2, n_samples))
    blocks_in = tr.split(audio_in, daw_bs, dim=1)
    blocks_out = []

    for block_in in blocks_in:
        if block_in.size(1) != daw_bs:
            break
        block_out = sqw.forward(block_in)
        block_out = tr.clone(block_out)
        blocks_out.append(block_out)

    audio_out = tr.cat(blocks_out, dim=1)

    actual_delay_l = tr.nonzero(audio_out[0, :])[0].item()
    actual_delay_r = tr.nonzero(audio_out[1, :])[0].item()
    assert actual_delay_l == actual_delay_r
    actual_delay = actual_delay_r
    assert expected_delay == actual_delay, (
        f"expected = {expected_delay}, actual_delay = {actual_delay} | "
        f"{daw_sr}, {daw_bs}, {model_sr}, {model_bs}"
    )


def test_calc_saturation_n() -> None:
    # random.seed(42)
    # tr.manual_seed(42)
    # io_buffer_sizes = [random.randrange(32, 2048) for _ in range(16)]
    # model_buffer_sizes = [random.randrange(32, 2048) for _ in range(16)]

    io_buffer_sizes = list(range(2, 256))
    model_buffer_sizes = list(range(2, 256))

    log.info(f"io_buffer_sizes: {io_buffer_sizes}")
    log.info(f"model_buffer_sizes: {model_buffer_sizes}")

    for io_bs, model_bs in tqdm(itertools.product(io_buffer_sizes, model_buffer_sizes)):
        calculated_n = SampleQueueWrapper.calc_saturation_n(io_bs, model_bs)
        found_n = find_saturation_n(io_bs, model_bs)
        assert found_n is not None, f"Could not find a saturation_n. io_bs = {io_bs}, model_bs = {model_bs}"
        assert found_n % io_bs == 0
        assert calculated_n == found_n, (
            f"io_bs = {io_bs}, model_bs = {model_bs}, calculated_n = {calculated_n}, found_n = {found_n}"
        )
        assert check_queue_saturation(io_bs, model_bs, found_n)

    log.info("No saturation inconsistencies found")


def test_calc_min_delay_samples() -> None:
    wrapper = TestModelWrapper()
    sqw = SampleQueueWrapper(wrapper)

    sampling_rates = [16000, 22050, 32000, 44100, 48000, 88200, 96000]
    buffer_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    # random.seed(42)
    # tr.manual_seed(42)
    # buffer_sizes = [random.randrange(32, 4096) for _ in range(50)]

    log.info(f"Sampling rates: {sampling_rates}")
    log.info(f"Buffer sizes: {buffer_sizes}")

    for daw_sr, daw_bs, model_sr, model_bs in tqdm(itertools.product(sampling_rates,
                                                                     buffer_sizes,
                                                                     sampling_rates,
                                                                     buffer_sizes)):
        delay_test(wrapper, sqw, daw_sr, daw_bs, model_sr, model_bs)

    log.info("No delay inconsistencies found")
