import logging
import math
import os
from typing import Optional, List

import torch as tr
from torch import Tensor, nn

from neutone_sdk import WaveformToWaveformMetadata
from neutone_sdk.constants import DEFAULT_DAW_SR, DEFAULT_DAW_BS
from neutone_sdk.queues import CircularInplaceTensorQueue
from neutone_sdk.sandwich import ChannelNormalizerSandwich, InplaceInterpolationResampler

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SampleQueueWrapper(nn.Module):
    def __init__(self,
                 w2w_base: "WaveformToWaveformBase",
                 daw_sr: int = DEFAULT_DAW_SR,
                 model_sr: Optional[int] = None,
                 daw_bs: int = DEFAULT_DAW_BS,
                 model_bs: Optional[int] = None,
                 use_debug_mode: bool = True) -> None:
        """
        Creates a SampleQueueWrapper (SQW) which wraps a WaveformToWaveformBase Neutone model to make it compatible
        with varying DAW sampling rates, buffer sizes, and also delay reporting, and multithreading.

        Related issue: https://github.com/QosmoInc/neutone_sdk/issues/6
        """
        super().__init__()
        self.use_debug_mode = use_debug_mode

        self.w2w_base = w2w_base
        self.in_n_ch = 1 if self.is_input_mono() else 2
        self.out_n_ch = 1 if self.is_output_mono() else 2

        self.channel_normalizer = ChannelNormalizerSandwich(use_debug_mode=use_debug_mode)
        # TODO(cm): switch to a more robust resampling method that prevents aliasing
        self.resample_sandwich = InplaceInterpolationResampler(self.in_n_ch,
                                                               self.out_n_ch,
                                                               daw_sr,
                                                               daw_sr,  # Tmp sample rate values
                                                               daw_bs,
                                                               use_debug_mode=use_debug_mode)
        self.params_resample_sandwich = InplaceInterpolationResampler(self.w2w_base.MAX_N_PARAMS,
                                                                      self.w2w_base.MAX_N_PARAMS,
                                                                      daw_sr,
                                                                      daw_sr,  # Tmp sample rate value
                                                                      daw_bs,
                                                                      use_debug_mode=use_debug_mode)

        self.daw_sr = daw_sr
        self.model_sr = model_sr
        self.daw_bs = daw_bs
        self.io_bs = daw_bs  # Temp value for typing
        self.model_bs = model_bs
        self.seen_n = 0
        self.is_queue_saturated = False
        self.saturation_n = None

        self.in_queue = None
        self.params_queue = None
        self.out_queue = None

        self.daw_buffer = None
        self.model_in_buffer = None
        self.params_buffer = None
        self.io_out_buffer = None
        self.bt_out_buffer = None

        self.set_daw_sample_rate_and_buffer_size(daw_sr, daw_bs, model_sr, model_bs)

    @staticmethod
    def select_best_model_sr(daw_sr: int, native_sample_rates: List[int]) -> int:
        """
        Given a DAW sampling rate and a list of all the sampling rates a Neutone model supports (usually only one, or
        an empty list indicates all sampling rates are supported), determine the optimal sampling rate to use.
        """
        # Avoid resampling whenever possible
        if not native_sample_rates:
            return daw_sr
        if daw_sr in native_sample_rates:
            return daw_sr
        # Resampling is unavoidable
        if len(native_sample_rates) == 1:
            return native_sample_rates[0]
        # TODO(cm): combine this with selecting the buffer size to be smarter
        # TODO(cm): prefer upsampling if the buffer sizes allow it
        # This is a workaround for torchscript not supporting lambda functions
        diffs = [abs(sr - daw_sr) for sr in native_sample_rates]
        min_idx = diffs.index(min(diffs))
        return native_sample_rates[min_idx]

    @staticmethod
    def select_best_model_buffer_size(io_bs: int, native_buffer_sizes: List[int]) -> int:
        """
        Given a DAW buffer size and a list of all the buffer sizes a Neutone model supports (usually only one, or
        an empty list indicates all buffer sizes are supported), determine the optimal buffer size to use.
        """
        if not native_buffer_sizes:
            return io_bs
        if len(native_buffer_sizes) == 1:
            return native_buffer_sizes[0]
        native_buffer_sizes = sorted(native_buffer_sizes)
        for bs in native_buffer_sizes:
            if bs % io_bs == 0:
                return bs
        for bs in native_buffer_sizes:
            if bs > io_bs:
                return bs
        # TODO(cm): prefer near bs // 2 if 0 padded forward passes are enabled
        # This is a workaround for torchscript not supporting lambda functions
        diffs = [abs(bs - io_bs) for bs in native_buffer_sizes]
        min_idx = diffs.index(min(diffs))
        return native_buffer_sizes[min_idx]

    @staticmethod
    def _calc_saturation_n_case_3(io_bs: int, model_bs: int) -> int:
        # TODO(cm): I cannot figure out an elegant formula for this specific case, but I'm sure it must exist
        # TODO(cm): this needs to be explained with a diagram / blog post
        io_bs_t = tr.tensor(io_bs)
        model_bs_t = tr.tensor(model_bs)
        lcm_t = tr.lcm(io_bs_t, model_bs_t)
        cycle_len = int(tr.div(lcm_t, io_bs_t, rounding_mode='trunc').item())  # TorchScript requires this casting
        remainders_shifted = (tr.arange(0, cycle_len, dtype=tr.int) * io_bs_t) % model_bs_t
        remainders = (tr.arange(1, cycle_len + 1, dtype=tr.int) * io_bs_t) % model_bs_t
        pop_locs = tr.where(remainders > remainders_shifted, 0, 1)
        pop_locs_rev = tr.flip(pop_locs, dims=(0,))
        pop_locs_cumsum = tr.cumsum(pop_locs, dim=0)
        pop_locs_rev_cumsum = tr.cumsum(pop_locs_rev, dim=0)
        offset = 0
        for _ in range(cycle_len):
            pop_locs_rev_cumsum_shifted = tr.zeros_like(pop_locs_rev_cumsum)
            pop_locs_rev_cumsum_shifted[offset:] = pop_locs_rev_cumsum[0:cycle_len - offset]
            if tr.all(pop_locs_cumsum >= pop_locs_rev_cumsum_shifted):
                break
            offset += 1
        return (offset + 1) * io_bs

    @staticmethod
    def calc_saturation_n(io_bs: int, model_bs: int) -> int:
        """
        Assume you have 2 queues. Every time `io_bs` samples are pushed onto queue 1, the same number of samples must be
        popped from queue 2. Whenever queue 1 contains `model_bs` samples or more, they are popped from queue 1 and
        pushed onto queue 2. This happens instantaneously after pushing to queue 1 and before popping from queue 2.
        A simple non-trivial example is when `io_bs` = 4 and `model_bs` = 7 (saturation_n is 12 not 8 in this case).

        This method calculates the minimum number of samples one must wait before popping from queue 2 to guarantee
        that it will never be starved (i.e. you cannot pop `io_bs` samples from queue 2). We call this `saturation_n`
        and it will always be a multiple of `io_bs` (since the best case scenario is you can pop immediately after the
        first buffer is pushed onto queue 1).
        """
        # TODO(cm): document logic behind this
        if model_bs % io_bs == 0:  # Case 1
            return model_bs
        if io_bs % model_bs == 0:  # Case 2
            return io_bs
        if io_bs < model_bs:  # Case 3
            return SampleQueueWrapper._calc_saturation_n_case_3(io_bs, model_bs)
        else:  # io_bs > model_bs, Case 4
            multiplier = io_bs // model_bs
            n = (multiplier * model_bs) + (io_bs % model_bs)
            return (n // io_bs + 1) * io_bs

    @staticmethod
    def calc_delay_samples(io_bs: int, model_bs: int) -> int:
        # TODO(cm): document logic behind this
        saturation_n = SampleQueueWrapper.calc_saturation_n(io_bs, model_bs)
        return saturation_n - io_bs

    @staticmethod
    def calc_resampled_buffer_size(orig_sr: int, new_sr: int, orig_bs: int) -> int:
        if orig_sr == new_sr:
            resampled_bs = orig_bs
        else:
            resampled_bs = int(math.ceil(new_sr * orig_bs / orig_sr))
        return resampled_bs

    @staticmethod
    def calc_max_daw_queue_size(daw_sr: int, daw_bs: int, model_sr: int, model_bs: int) -> int:
        daw_model_bs = int(model_bs * daw_sr / model_sr) + 1
        return (2 * daw_bs) + daw_model_bs

    def prepare_for_inference(self) -> None:
        self.w2w_base.prepare_for_inference()
        self.use_debug_mode = False
        self.channel_normalizer.use_debug_mode = False
        self.resample_sandwich.use_debug_mode = False
        self.params_resample_sandwich.use_debug_mode = False
        self.in_queue.use_debug_mode = False
        self.params_queue.use_debug_mode = False
        self.out_queue.use_debug_mode = False
        self.eval()

    def _forward(self, resampled_x: Tensor, params: Optional[Tensor] = None) -> None:
        if params is not None:
            params = self.params_resample_sandwich.process_in(params)
            self.params_queue.push(params)

        resampled_in_n = resampled_x.size(1)
        if self.use_debug_mode:
            assert resampled_in_n == self.io_bs

        self.in_queue.push(resampled_x)
        if not self.is_queue_saturated:
            self.seen_n += resampled_in_n
            if self.seen_n >= self.saturation_n:
                self.is_queue_saturated = True

        while self.in_queue.size >= self.model_bs:
            in_popped_n = self.in_queue.pop(self.model_in_buffer)
            if self.use_debug_mode:
                assert in_popped_n == self.model_bs

            if self.params_queue.is_empty():
                model_out = self.w2w_base.forward(self.model_in_buffer, None)
            else:
                params_popped_n = self.params_queue.pop(self.params_buffer)
                if self.use_debug_mode:
                    assert params_popped_n == in_popped_n
                model_out = self.w2w_base.forward(self.model_in_buffer, self.params_buffer)

            self.out_queue.push(model_out)

    @tr.no_grad()
    def forward(self, x: Tensor, params: Optional[Tensor] = None) -> Tensor:
        is_daw_mono = x.size(0) == 1
        in_n = x.shape[1]
        x = self.channel_normalizer(x, self.is_input_mono(), self.daw_buffer)
        x = self.resample_sandwich.process_in(x)
        self._forward(x, params)

        if self.is_queue_saturated:
            out_popped_n = self.out_queue.pop(self.io_out_buffer)
        else:
            out_popped_n = self.io_out_buffer.size(1)
            self.io_out_buffer.fill_(0)

        x = self.resample_sandwich.process_out(self.io_out_buffer)
        if self.use_debug_mode:
            assert out_popped_n == self.io_out_buffer.size(1)
            assert x.size(1) == in_n
        x = self.channel_normalizer(x, is_daw_mono, self.daw_buffer)
        return x

    @tr.jit.export
    @tr.no_grad()
    def forward_bt(self, x: Tensor, params: Optional[Tensor] = None) -> Optional[Tensor]:
        daw_n_ch = x.size(0)
        is_daw_mono = daw_n_ch == 1
        x = self.channel_normalizer(x, self.is_input_mono(), self.daw_buffer)
        x = self.resample_sandwich.process_in(x)
        self._forward(x, params)

        curr_n = 0
        while self.is_queue_saturated and self.out_queue.size >= self.io_bs:
            out_popped_n = self.out_queue.pop(self.io_out_buffer)
            if self.use_debug_mode:
                assert out_popped_n == self.io_bs
            x = self.resample_sandwich.process_out(self.io_out_buffer)
            x = self.channel_normalizer(x, is_daw_mono, self.daw_buffer)
            if self.use_debug_mode:
                assert x.size(1) == self.daw_bs
                assert curr_n + self.daw_bs <= self.bt_out_buffer.size(1)
            self.bt_out_buffer[0:daw_n_ch, curr_n:curr_n + self.daw_bs] = x
            curr_n += self.daw_bs

        if curr_n == 0:
            return None
        return self.bt_out_buffer[0:daw_n_ch, 0:curr_n]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return self.w2w_base.is_input_mono()

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return self.w2w_base.is_output_mono()

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return self.w2w_base.get_native_sample_rates()

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return self.w2w_base.get_native_buffer_sizes()

    @tr.jit.export
    def is_resampling(self) -> bool:
        return self.resample_sandwich.is_resampling()

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        model_min_delay = self.w2w_base.calc_min_delay_samples()
        wrapper_min_delay = self.calc_delay_samples(self.io_bs, self.model_bs)
        min_delay = model_min_delay + wrapper_min_delay
        if self.is_resampling():
            min_delay = int(min_delay * self.daw_bs / self.io_bs)

        return min_delay

    @tr.jit.export
    def set_daw_sample_rate_and_buffer_size(
            self,
            daw_sr: int,
            daw_bs: int,
            model_sr: Optional[int] = None,
            model_bs: Optional[int] = None,
    ) -> int:
        # Sample rate
        if model_sr is not None:
            if self.use_debug_mode:
                assert len(self.get_native_sample_rates()) == 0 or model_sr in self.get_native_sample_rates()
        else:
            model_sr = self.select_best_model_sr(daw_sr, self.get_native_sample_rates())

        io_bs = self.calc_resampled_buffer_size(daw_sr, model_sr, daw_bs)

        self.resample_sandwich.set_sample_rates(daw_sr, model_sr, daw_bs)
        self.params_resample_sandwich.set_sample_rates(daw_sr, model_sr, daw_bs)
        self.daw_sr = daw_sr
        self.model_sr = model_sr

        # Buffer size
        if model_bs is not None:
            if self.use_debug_mode:
                assert len(self.get_native_buffer_sizes()) == 0 or model_bs in self.get_native_buffer_sizes()
        else:
            model_bs = self.select_best_model_buffer_size(io_bs, self.get_native_buffer_sizes())

        self.w2w_base.set_buffer_size(model_bs)
        self.daw_bs = daw_bs
        self.io_bs = io_bs
        self.model_bs = model_bs

        self.in_queue = CircularInplaceTensorQueue(self.in_n_ch,
                                                   (2 * self.io_bs) + self.model_bs,
                                                   use_debug_mode=self.use_debug_mode)
        self.params_queue = CircularInplaceTensorQueue(self.w2w_base.MAX_N_PARAMS,
                                                       (2 * self.io_bs) + self.model_bs,
                                                       use_debug_mode=self.use_debug_mode)
        self.out_queue = CircularInplaceTensorQueue(self.out_n_ch,
                                                    (2 * self.io_bs) + self.model_bs,
                                                    use_debug_mode=self.use_debug_mode)

        self.daw_buffer = tr.zeros((2, self.daw_bs))
        self.model_in_buffer = tr.zeros((self.in_n_ch, self.model_bs))
        self.params_buffer = tr.zeros((self.w2w_base.MAX_N_PARAMS, self.model_bs))
        self.io_out_buffer = tr.zeros((self.out_n_ch, self.io_bs))

        max_daw_queue_size = self.calc_max_daw_queue_size(self.daw_sr,
                                                          self.daw_bs,
                                                          self.model_sr,
                                                          self.model_bs)
        self.bt_out_buffer = tr.zeros((2, max_daw_queue_size))

        self.saturation_n = self.calc_saturation_n(self.io_bs, self.model_bs)
        self.reset()

        return max_daw_queue_size

    @tr.jit.export
    def reset(self) -> None:
        self.w2w_base.reset()
        self.in_queue.reset()
        self.params_queue.reset()
        self.out_queue.reset()
        self.daw_buffer.fill_(0)
        self.model_in_buffer.fill_(0)
        self.params_buffer.fill_(0)
        self.io_out_buffer.fill_(0)
        self.bt_out_buffer.fill_(0)
        self.seen_n = 0
        self.is_queue_saturated = False

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        return [
            "forward_bt",
            "is_input_mono",
            "is_output_mono",
            "get_native_sample_rates",
            "get_native_buffer_sizes",
            "is_resampling",
            "calc_min_delay_samples",
            "set_daw_sample_rate_and_buffer_size",
            "reset",
            "get_preserved_attributes",
            "to_metadata",
            "w2w_base",
        ]

    @tr.jit.export
    def to_metadata(self) -> WaveformToWaveformMetadata:
        return self.w2w_base.to_metadata()
