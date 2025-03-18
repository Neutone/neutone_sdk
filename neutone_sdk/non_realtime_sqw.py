import logging
import math
import os
from typing import Dict, List, Optional, Any

import torch as tr
from torch import Tensor, nn
from torch import Tensor as T

from neutone_sdk import (
    DEFAULT_DAW_BS,
    DEFAULT_DAW_SR,
    ChannelNormalizerSandwich,
    Inplace4pHermiteResampler,
    InplaceLinearResampler,
    utils,
)
from neutone_sdk.non_realtime_wrapper import NonRealtimeBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NonRealtimeSampleQueueWrapper(nn.Module):
    def __init__(
        self,
        nrb: NonRealtimeBase,
        daw_sr: int = DEFAULT_DAW_SR,
        daw_bs: int = DEFAULT_DAW_BS,
        use_debug_mode: bool = True,
    ) -> None:
        """
        Wraps a PyTorch model for use in a non-realtime context.
        Compatible with the Neutone Gen plugin.
        """
        super().__init__()
        self.nrb = nrb
        self.daw_sr = daw_sr
        self.use_debug_mode = use_debug_mode

        self.n_in_tracks = len(self.get_audio_in_channels())
        self.n_out_tracks = len(self.get_audio_out_channels())
        self.model_sr = utils.select_best_model_sr(
            self.daw_sr, self.get_native_sample_rates()
        )

        self.channel_normalizer = ChannelNormalizerSandwich(
            use_debug_mode=use_debug_mode
        )
        self.resample_sandwich_mono = Inplace4pHermiteResampler(
            in_n_ch=1,
            out_n_ch=1,
            in_sr=self.daw_sr,
            out_sr=self.model_sr,  # Tmp sample rate values
            in_bs=daw_bs,
            use_debug_mode=use_debug_mode,
        )
        self.resample_sandwich_stereo = Inplace4pHermiteResampler(
            in_n_ch=2,
            out_n_ch=2,
            in_sr=self.daw_sr,
            out_sr=self.model_sr,  # Tmp sample rate values
            in_bs=daw_bs,
            use_debug_mode=use_debug_mode,
        )
        self.params_resample_sandwich = InplaceLinearResampler(
            in_n_ch=self.nrb.n_numerical_params,
            out_n_ch=self.nrb.n_numerical_params,
            in_sr=self.daw_sr,
            out_sr=self.model_sr,  # Tmp sample rate value
            in_bs=daw_bs,
            use_debug_mode=use_debug_mode,
        )

    def prepare_for_inference(self) -> None:
        self.nrb.prepare_for_inference()
        self.use_debug_mode = False
        self.channel_normalizer.use_debug_mode = False
        self.resample_sandwich_mono.use_debug_mode = False
        self.resample_sandwich_stereo.use_debug_mode = False
        self.params_resample_sandwich.use_debug_mode = False
        self.eval()

    @tr.jit.export
    def get_audio_in_channels(self) -> List[int]:
        return self.nrb.get_audio_in_channels()

    @tr.jit.export
    def get_audio_out_channels(self) -> List[int]:
        return self.nrb.get_audio_out_channels()

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return self.nrb.get_native_sample_rates()

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return self.nrb.get_native_buffer_sizes()

    @tr.jit.export
    def is_resampling(self) -> bool:
        return self.resample_sandwich_mono.is_resampling()

    @tr.jit.export
    def set_daw_sample_rate_and_buffer_size(
        self,
        daw_sr: int,
        daw_bs: int,
        model_sr: Optional[int] = None,
        model_bs: Optional[int] = None,
    ) -> int:
        # Only sample rate is adjusted here, but we keep the interface of the SQW
        if model_sr is not None:
            if self.use_debug_mode:
                assert (
                    len(self.get_native_sample_rates()) == 0
                    or model_sr in self.get_native_sample_rates()
                )
        else:
            model_sr = utils.select_best_model_sr(
                daw_sr, self.get_native_sample_rates()
            )
        self.daw_sr = daw_sr
        self.model_sr = model_sr
        self.reset()
        return -1  # We return -1 just to match the interface of the SQW

    @tr.jit.export
    def is_one_shot_model(self) -> bool:
        return self.nrb.is_one_shot_model()

    def forward(
        self,
        audio_in: List[Tensor],
        numerical_params: Optional[Tensor] = None,
        text_params: Optional[List[str]] = None,
    ) -> List[Tensor]:
        if self.use_debug_mode:
            assert len(audio_in) == self.n_in_tracks
            in_n_samples = 0
            for in_ch, in_track in zip(self.get_audio_in_channels(), audio_in):
                assert 1 <= in_ch <= 2
                if in_n_samples == 0:
                    in_n_samples = in_track.size(1)
                    assert in_n_samples > 0
                else:
                    assert in_n_samples == in_track.size(1)

        # Normalize channels and resample input audio
        audio_in_proc = []
        in_n_samples = 0
        in_n_samples_proc = 0
        if audio_in:
            in_n_samples = audio_in[0].size(1)
            # Setup audio resamplers
            self.resample_sandwich_mono.set_sample_rates(
                self.daw_sr, self.model_sr, in_n_samples
            )
            self.resample_sandwich_stereo.set_sample_rates(
                self.daw_sr, self.model_sr, in_n_samples
            )
            for in_ch, in_track in zip(self.get_audio_in_channels(), audio_in):
                if self.should_cancel_forward_pass():
                    return []

                # The channel normalizer doesn't allocate memory so we need this
                out = tr.zeros((2, in_n_samples), dtype=in_track.dtype)
                should_be_mono = in_ch == 1
                in_track_proc = self.channel_normalizer(in_track, should_be_mono, out)
                # Resample input audio
                if self.is_resampling():
                    if should_be_mono:
                        in_track_proc = self.resample_sandwich_mono.process_in(
                            in_track_proc
                        )
                    else:
                        in_track_proc = self.resample_sandwich_stereo.process_in(
                            in_track_proc
                        )
                    # The resampler doesn't allocate memory so we need to clone the
                    # output in case there are multiple input audio tracks
                    in_track_proc = in_track_proc.clone()

                audio_in_proc.append(in_track_proc)

                # Update in_n_samples_proc value during the first iteration
                if in_n_samples_proc == 0:
                    in_n_samples_proc = in_track_proc.size(1)

                if self.use_debug_mode:
                    assert in_track_proc.size(0) == in_ch
                    assert in_track_proc.size(1) == in_n_samples_proc

        if self.should_cancel_forward_pass():
            return []

        # Resample parameters
        if numerical_params is not None:
            if self.use_debug_mode:
                assert numerical_params.size(0) == self.nrb.n_numerical_params
                if in_n_samples > 0:
                    assert numerical_params.size(1) == in_n_samples
            if in_n_samples == 0:
                in_n_samples = numerical_params.size(1)
            # Setup params resampler
            self.params_resample_sandwich.set_sample_rates(
                self.daw_sr, self.model_sr, in_n_samples
            )
            if self.is_resampling():
                numerical_params = self.params_resample_sandwich.process_in(
                    numerical_params
                )
            # Update in_n_samples_proc value if it hasn't been done yet
            if in_n_samples_proc == 0:
                in_n_samples_proc = numerical_params.size(1)

            if self.use_debug_mode:
                assert numerical_params.size(0) == self.nrb.n_numerical_params
                assert numerical_params.size(1) == in_n_samples_proc

        if self.should_cancel_forward_pass():
            return []

        # Prepare blocks if needed
        n_blocks = 1
        # This is in the model's sample rate which is important
        model_delay = self.nrb.calc_model_delay_samples()
        # We pad by at least the delay amount to ensure we have enough samples
        delay_padding = model_delay
        block_padding = 0
        audio_in_blocks = []

        # Pad and unfold input audio
        if audio_in_proc:
            # Calculate model_bs
            model_bs = utils.select_best_model_buffer_size(
                in_n_samples_proc + model_delay, self.get_native_buffer_sizes()
            )
            if not self.is_one_shot_model():
                n_blocks = math.ceil((in_n_samples_proc + delay_padding) / model_bs)
                block_padding = n_blocks * model_bs - in_n_samples_proc - delay_padding
            for in_track_proc in audio_in_proc:
                in_track_proc = tr.nn.functional.pad(
                    in_track_proc,
                    (0, delay_padding + block_padding),
                    mode="constant",
                    value=0.0,
                )
                if n_blocks > 1:
                    in_track_blocks = in_track_proc.unfold(1, model_bs, model_bs)
                else:
                    in_track_blocks = in_track_proc.unsqueeze(1)
                audio_in_blocks.append(in_track_blocks)

        if self.should_cancel_forward_pass():
            return []

        # Pad and unfold input parameters
        numerical_params_blocks: Optional[T] = None
        if numerical_params is not None:
            # Calculate model_bs
            model_bs = utils.select_best_model_buffer_size(
                in_n_samples_proc + delay_padding, self.get_native_buffer_sizes()
            )
            if not self.is_one_shot_model():
                n_blocks = math.ceil((in_n_samples_proc + delay_padding) / model_bs)
                block_padding = n_blocks * model_bs - in_n_samples_proc - delay_padding
            numerical_params = tr.nn.functional.pad(
                numerical_params, (0, delay_padding + block_padding), mode="replicate"
            )
            if n_blocks > 1:
                numerical_params_blocks = numerical_params.unfold(1, model_bs, model_bs)
            else:
                numerical_params_blocks = numerical_params.unsqueeze(1)

        if self.should_cancel_forward_pass():
            return []

        # Inference
        audio_out_blocks: List[List[T]] = []
        numerical_params_block: Optional[T] = None
        for idx in range(n_blocks):
            if self.should_cancel_forward_pass():
                return []

            audio_in_block = [b[:, idx, :] for b in audio_in_blocks]
            if numerical_params_blocks is not None:
                numerical_params_block = numerical_params_blocks[:, idx, :]
            audio_out_block = self.nrb.forward(
                idx, audio_in_block, numerical_params_block, text_params
            )
            audio_out_blocks.append(audio_out_block)

        # Concat and remove padding from output audio
        audio_out = []
        out_n_samples = 0
        for track_idx in range(self.n_out_tracks):
            if self.should_cancel_forward_pass():
                return []

            # Create track from blocks, we can't use zip(*) here because of TorchScript
            track_blocks = []
            for block_idx in range(n_blocks):
                track_blocks.append(audio_out_blocks[block_idx][track_idx])
            track = tr.cat(track_blocks, dim=1)

            # Remove delay padding if needed
            curr_n_samples = track.size(1)
            if 0 < delay_padding < curr_n_samples:
                track = track[:, delay_padding:]

            # Remove block padding if needed
            curr_n_samples = track.size(1)
            if 0 < block_padding < curr_n_samples:
                track = track[:, :-block_padding]

            # Update out_n_samples value during the first iteration
            if out_n_samples == 0:
                out_n_samples = track.size(1)

            if self.use_debug_mode:
                out_n_ch = self.get_audio_out_channels()[track_idx]
                assert track.size(0) == out_n_ch
                assert track.size(1) == out_n_samples

            audio_out.append(track)

        # Resample output audio
        audio_out_proc = []
        # We need to reconfigure the resamplers if the output audio size has changed
        # and use process_in instead of process_out
        reconfigured_resamplers = False
        if self.resample_sandwich_mono.out_bs != out_n_samples:
            self.resample_sandwich_mono.set_sample_rates(
                self.model_sr, self.daw_sr, out_n_samples
            )
            self.resample_sandwich_stereo.set_sample_rates(
                self.model_sr, self.daw_sr, out_n_samples
            )
            reconfigured_resamplers = True

        if self.is_resampling():
            # Resample output audio
            for out_track in audio_out:
                if self.should_cancel_forward_pass():
                    return []

                out_ch = out_track.size(0)
                if out_ch == 1:
                    if reconfigured_resamplers:
                        out_track_proc = self.resample_sandwich_mono.process_in(
                            out_track
                        )
                    else:
                        out_track_proc = self.resample_sandwich_mono.process_out(
                            out_track
                        )
                else:
                    if reconfigured_resamplers:
                        out_track_proc = self.resample_sandwich_stereo.process_in(
                            out_track
                        )
                    else:
                        out_track_proc = self.resample_sandwich_stereo.process_out(
                            out_track
                        )
                # The resampler doesn't allocate memory so we need to clone the output
                # in case there are multiple output audio tracks
                out_track_proc = out_track_proc.clone()
                audio_out_proc.append(out_track_proc)
        else:
            audio_out_proc = audio_out

        return audio_out_proc

    @tr.jit.export
    def get_progress_percentage(self) -> int:
        return self.nrb.get_progress_percentage()

    @tr.jit.export
    def should_cancel_forward_pass(self) -> bool:
        return self.nrb.should_cancel_forward_pass()

    @tr.jit.export
    def request_cancel_forward_pass(self) -> None:
        self.nrb.request_cancel_forward_pass()

    @tr.jit.export
    def is_text_model(self) -> bool:
        return self.nrb.is_text_model()

    @tr.jit.export
    def reset(self) -> None:
        self.nrb.reset()

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        return [
            "nrb",
            "get_audio_in_channels",
            "get_audio_out_channels",
            "get_native_sample_rates",
            "get_native_buffer_sizes",
            "is_resampling",
            "set_daw_sample_rate_and_buffer_size",
            "is_one_shot_model",
            "get_progress_percentage",
            "should_cancel_forward_pass",
            "request_cancel_forward_pass",
            "is_text_model",
            "reset",
            "get_preserved_attributes",
            "to_metadata",
            "get_metadata_json",
            "get_default_param_values",
        ]

    @tr.jit.export
    def to_metadata(self) -> Dict[str, Any]:
        return self.nrb.to_metadata()

    @tr.jit.export
    def get_metadata_json(self) -> str:
        return self.nrb.get_metadata_json()

    @tr.jit.export
    def get_default_param_values(self) -> Tensor:
        return self.nrb.get_default_param_values()
