import logging
import os
from typing import Optional, List

import torch as tr
from torch import Tensor
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


# TODO(cm:) add documentation, for now please see the documentation in `examples/example_spectral_filter.py`
class RealtimeSTFT(nn.Module):
    def __init__(
        self,
        model_io_n_frames: int = 16,
        io_n_ch: int = 2,
        io_n_samples: int = 512,
        n_fft: int = 2048,
        hop_len: int = 512,
        window: Optional[Tensor] = None,
        center: bool = True,
        power: Optional[float] = 1.0,
        logarithmize: bool = True,
        ensure_pos_spec: bool = True,
        use_phase_info: bool = True,
        fade_n_samples: int = 0,
        eps: float = 1e-8,
        use_debug_mode: bool = True
    ) -> None:
        super().__init__()
        self.use_debug_mode = use_debug_mode
        if self.use_debug_mode:
            assert n_fft % 2 == 0
            assert (n_fft // 2) % hop_len == 0
            if window is not None:
                assert window.shape == (n_fft,)
            assert center, "Behavior of center=False needs to be debugged, results in artefacts"
            # if center:
            #     log.warning("STFT is not causal when center=True")
            assert power is None or power >= 1.0
            if power is None and use_phase_info:
                log.warning("If power=None, `use_phase_info=True` means the imag component is saved, not the angle")
            if power is not None and power > 1.0:
                log.warning(
                    "A power greater than 1.0 probably adds unnecessary "
                    "computational complexity"
                )
            assert fade_n_samples < io_n_samples
        self.model_io_n_frames = model_io_n_frames
        self.io_n_ch = io_n_ch
        self.io_n_samples = io_n_samples
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.center = center
        self.power = power
        self.logarithmize = logarithmize
        self.ensure_pos_spec = ensure_pos_spec
        self.use_phase_info = use_phase_info
        self.fade_n_samples = fade_n_samples
        self.eps = eps

        # Derived parameters
        self.io_n_frames = None
        self.overlap_n_frames = None
        self.in_buf_n_frames = None
        self.n_bins = None
        self.stft_out_shape = None
        self.istft_in_n_frames = None
        self.istft_length = None
        self.model_io_shape = None
        self.out_buf_n_samples = None

        # Internal buffers
        self.in_buf = None
        self.in_buf_tmp = None
        self.stft_mag_buf = None
        self.mag_buf = None
        self.mag_buf_tmp = None
        self.spec_out_buf = None
        self.stft_phase_buf = None
        self.phase_buf = None
        self.phase_buf_tmp = None
        self.out_frames_buf = None
        self.out_buf = None

        # Sets derived parameters and allocates buffers
        self.set_buffer_size(io_n_samples)

        # Internal tensors
        if window is None:
            window = tr.hann_window(self.n_fft)
            if not center:
                # Ensures the NOLA constraint is met for the hann_window
                # See https://github.com/pytorch/pytorch/issues/62323
                # 1e-5 is chosen based on the torchaudio implementation
                window = tr.clamp(window, min=1e-5)
        self.register_buffer("window", window, persistent=True)
        log10_eps = tr.log10(tr.tensor([self.eps]))
        self.register_buffer("log10_eps", log10_eps, persistent=False)
        fade_up = tr.linspace(0, 1, max(self.fade_n_samples, 1))
        self.register_buffer("fade_up", fade_up, persistent=False)
        fade_down = tr.linspace(1, 0, max(self.fade_n_samples, 1))
        self.register_buffer("fade_down", fade_down, persistent=False)
        zero_phase = tr.zeros(self.model_io_shape)
        self.register_buffer("zero_phase", zero_phase, persistent=False)
        self.ten_constant = tr.tensor(10.0)

    def _set_derived_params(self) -> None:
        self.io_n_frames = self.io_n_samples // self.hop_len
        if self.use_debug_mode:
            assert self.io_n_frames <= self.model_io_n_frames
        self.overlap_n_frames = self.n_fft // 2 // self.hop_len
        self.in_buf_n_frames = (2 * self.overlap_n_frames) + self.io_n_frames - 1
        self.n_bins = (self.n_fft // 2) + 1
        if self.center:
            self.stft_out_shape = (
                self.io_n_ch,
                self.n_bins,
                (2 * self.overlap_n_frames) + self.io_n_frames,
            )
            self.istft_in_n_frames = self.overlap_n_frames + self.io_n_frames
            self.istft_length = (self.istft_in_n_frames - 1) * self.hop_len
        else:
            self.stft_out_shape = (self.io_n_ch, self.n_bins, self.io_n_frames)
            self.istft_in_n_frames = self.io_n_frames
            self.istft_length = self.in_buf_n_frames * self.hop_len
        if self.use_debug_mode:
            assert self.istft_in_n_frames <= self.model_io_n_frames

        self.model_io_shape = (self.io_n_ch, self.n_bins, self.model_io_n_frames)
        self.out_buf_n_samples = self.io_n_samples + self.fade_n_samples
        if self.use_debug_mode:
            assert self.out_buf_n_samples <= self.istft_length

    def _allocate_buffers(self) -> None:
        self.in_buf = tr.full(
            (self.io_n_ch, self.in_buf_n_frames * self.hop_len),
            self.eps,
        )
        self.in_buf_tmp = tr.clone(self.in_buf)

        self.stft_mag_buf = tr.full(self.stft_out_shape, self.eps)
        self.mag_buf = tr.full(self.model_io_shape, self.eps)
        self.mag_buf_tmp = tr.clone(self.mag_buf)
        # Required to allow inplace operations after the encoder
        self.spec_out_buf = tr.clone(self.mag_buf)

        self.stft_phase_buf = tr.zeros(self.stft_out_shape)
        self.phase_buf = tr.zeros(self.model_io_shape)
        self.phase_buf_tmp = tr.clone(self.phase_buf)

        self.out_frames_buf = tr.full(
            (self.io_n_ch, self.n_bins, self.istft_in_n_frames),
            self.eps,
            dtype=tr.complex64,
        )
        self.out_buf = tr.full(
            (self.io_n_ch, self.out_buf_n_samples),
            self.eps,
        )

    def _logarithmize_spec(self, spec: Tensor) -> None:
        tr.clamp(spec, min=self.eps, out=spec)
        tr.log10(spec, out=spec)

    def _unlogarithmize_spec(self, spec: Tensor) -> None:
        tr.pow(self.ten_constant, spec, out=spec)
        tr.clamp(spec, min=self.eps, out=spec)

    def _update_mag_or_phase_buffers(
        self, stft_out_buf: Tensor, frames_buf: Tensor, frames_buf_tmp: Tensor
    ) -> None:
        if self.center:
            # Remove overlap frames we have computed before
            frames = stft_out_buf[:, :, self.overlap_n_frames :]
            # Identify frames that are more correct due to missing prev audio
            fixed_prev_frames = frames[:, :, : -self.io_n_frames]
            if self.use_debug_mode:
                assert fixed_prev_frames.size(2) == self.overlap_n_frames
            # Identify the new frames for the input audio chunk
            new_frames = frames[:, :, -self.io_n_frames :]
            # Overwrite previous frames with more correct frames
            n_fixed_frames = min(self.model_io_n_frames, self.overlap_n_frames)
            frames_buf[:, :, -n_fixed_frames:] = fixed_prev_frames[
                :, :, -n_fixed_frames:
            ]
        else:
            new_frames = stft_out_buf[:, :, -self.io_n_frames :]

        # Shift buffer left and insert new frames (this way because tr.roll allocates memory dynamically)
        frames_buf_tmp[:, :, :-self.io_n_frames] = frames_buf[:, :, self.io_n_frames:]
        frames_buf[:, :, :-self.io_n_frames] = frames_buf_tmp[:, :, :-self.io_n_frames]
        frames_buf[:, :, -self.io_n_frames:] = new_frames

    @tr.jit.export
    def set_buffer_size(self, io_n_samples: int) -> None:
        if self.use_debug_mode:
            assert io_n_samples >= self.hop_len
            assert io_n_samples % self.hop_len == 0
            assert self.fade_n_samples <= io_n_samples
        self.io_n_samples = io_n_samples
        self._set_derived_params()
        self._allocate_buffers()
        self.reset()

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        return self.fade_n_samples

    @tr.jit.export
    def reset(self) -> None:
        self.in_buf.fill_(self.eps)
        self.stft_mag_buf.fill_(self.eps)
        self.mag_buf.fill_(self.eps)
        self.spec_out_buf.fill_(self.eps)
        self.stft_phase_buf.fill_(0)
        self.phase_buf.fill_(0)
        self.out_frames_buf.fill_(self.eps)
        self.out_buf.fill_(self.eps)

    @tr.jit.export
    def calc_min_buffer_size(self) -> int:
        return self.hop_len

    @tr.jit.export
    def calc_max_buffer_size(self) -> int:
        return self.model_io_n_frames * self.hop_len

    @tr.jit.export
    def calc_supported_buffer_sizes(self) -> List[int]:
        min_buffer_size = self.calc_min_buffer_size()
        max_buffer_size = self.calc_max_buffer_size()
        buffer_sizes = [
            bs for bs in range(min_buffer_size, max_buffer_size + 1, self.hop_len)
        ]
        return buffer_sizes

    @tr.jit.ignore
    def audio_to_spec_offline(self, audio: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert audio.size(0) == self.io_n_ch
            assert audio.size(1) >= self.n_fft
            assert audio.size(1) % self.hop_len == 0
        spec = tr.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        if self.power is None:
            spec = spec.real
        else:
            spec = spec.abs()
            if self.power != 1.0:
                spec = spec.pow(self.power)

        if self.logarithmize:
            spec = tr.clamp(spec, min=self.eps)
            spec = tr.log10(spec)
            if self.ensure_pos_spec:
                spec -= self.log10_eps

        return spec

    @tr.jit.export
    def audio_to_spec(self, audio: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert audio.shape == (self.io_n_ch, self.io_n_samples)
        # Shift buffer left and insert audio chunk (this way because tr.roll allocates memory dynamically)
        self.in_buf_tmp[:, :-self.io_n_samples] = self.in_buf[:, self.io_n_samples:]
        self.in_buf[:, :-self.io_n_samples] = self.in_buf_tmp[:, :-self.io_n_samples]
        self.in_buf[:, -self.io_n_samples:] = audio

        # TODO(cm): allow pad_mode to be selected
        complex_frames = tr.stft(
            self.in_buf,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        if self.power is None:
            self.stft_mag_buf = complex_frames.real
        else:
            tr.abs(complex_frames, out=self.stft_mag_buf)
            if self.power != 1.0:
                tr.pow(self.stft_mag_buf, self.power, out=self.stft_mag_buf)
        if self.logarithmize:
            self._logarithmize_spec(self.stft_mag_buf)
            if self.ensure_pos_spec:
                self.stft_mag_buf -= self.log10_eps

        self._update_mag_or_phase_buffers(self.stft_mag_buf, self.mag_buf, self.mag_buf_tmp)

        if self.use_phase_info:
            if self.power is None:
                self.stft_phase_buf = complex_frames.imag
            else:
                tr.angle(complex_frames, out=self.stft_phase_buf)
            self._update_mag_or_phase_buffers(self.stft_phase_buf, self.phase_buf, self.phase_buf_tmp)

        # Prevent future inplace operations from mutating self.mag_buf
        self.spec_out_buf[:, :] = self.mag_buf
        return self.spec_out_buf

    @tr.jit.export
    def spec_to_audio(self, spec: Tensor) -> Tensor:
        if self.use_debug_mode:
            assert spec.shape == self.model_io_shape
        spec = spec[:, :, -self.istft_in_n_frames :]
        if self.use_phase_info:
            phase = self.phase_buf[:, :, -self.istft_in_n_frames:]
        else:
            phase = self.zero_phase[:, :, -self.istft_in_n_frames:]

        if self.logarithmize:
            if self.ensure_pos_spec:
                spec += self.log10_eps
            self._unlogarithmize_spec(spec)

        if self.power is None:
            self.out_frames_buf.real = spec
            self.out_frames_buf.imag = phase
        else:
            if self.power != 1.0:
                tr.pow(spec, 1 / self.power, out=spec)
            tr.polar(spec, phase, out=self.out_frames_buf)

        # TODO(cm): allow pad_mode to be selected
        rec_audio = tr.istft(
            self.out_frames_buf,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            window=self.window,
            center=self.center,
            length=self.istft_length,
        )
        rec_audio = rec_audio[:, -self.out_buf_n_samples:]
        if self.fade_n_samples == 0:
            return rec_audio

        self.out_buf[:, -self.fade_n_samples:] *= self.fade_down
        rec_audio[:, : self.fade_n_samples] *= self.fade_up
        rec_audio[:, : self.fade_n_samples] += self.out_buf[:, -self.fade_n_samples :]
        audio_out = rec_audio[:, : self.io_n_samples]
        self.out_buf = rec_audio
        return audio_out
