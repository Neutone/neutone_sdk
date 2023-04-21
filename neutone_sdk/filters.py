import math
from typing import List, Optional
from enum import Enum
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Filters for pre-filtering inputs to models such as RAVE.
"""


class FilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class FIRFilter(nn.Module):
    def __init__(
        self,
        filt_type: FilterType,
        cutoffs: List[float],
        filt_size: int = 257,
    ):
        """Streamable FIR filter for pre-filtering of model inputs, etc.

        Args:
            filt_type (FilterType): Type of the filter (FilterType.LOWPASS/HIGHPASS/BANDPASS/BANDSTOP).
            cutoffs (List[float]): Cutoff frequencies (in Hz). 2 should be given if bandpass/stop
            sample_rate (int): Sampling rate
            filt_size (int, optional): Length of the FIR. Defaults to 257.
        """
        super().__init__()
        # register buffer only allowed once
        self.register_buffer("cache", torch.zeros(2, filt_size - 1))
        self.register_buffer("ir_windowed", torch.empty(1, 1, filt_size))
        # Pass in fake sample rate for filter
        # Sample rate should be automatically overwritten by calling
        # set_parameters() from w2wbase.set_model_sample_rate_and_buffer_size()
        self.set_parameters(filt_type, cutoffs, 48000, filt_size)

    def set_parameters(
        self,
        filt_type: Optional[FilterType] = None,
        cutoffs: Optional[List[float]] = None,
        sample_rate: Optional[int] = None,
        filt_size: Optional[int] = None,
    ):
        filt_type = self.filt_type if filt_type is None else filt_type
        cutoffs = self.cutoffs if cutoffs is None else cutoffs
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        filt_size = self.filt_size if filt_size is None else filt_size
        if len(cutoffs) == 2:
            if filt_type.value in [FilterType.HIGHPASS.value, FilterType.LOWPASS.value]:
                raise ValueError(
                    f"only 1 cutoff value supported for filter type: {filt_type}"
                )
        else:
            if filt_type.value in [
                FilterType.BANDPASS.value,
                FilterType.BANDSTOP.value,
            ]:
                raise ValueError(
                    f"2 cutoff values (low, high) needed for filter type: {filt_type}"
                )
        # create frequency response by frequency sampling
        freqs = torch.fft.rfftfreq(filt_size, 1 / sample_rate)

        if filt_type == FilterType.HIGHPASS:
            freq_resp = torch.where((freqs > cutoffs[0]), 1.0, 0.0).float()
        elif filt_type == FilterType.LOWPASS:
            freq_resp = torch.where((freqs < cutoffs[0]), 1.0, 0.0).float()
        elif filt_type == FilterType.BANDPASS:
            freq_resp = torch.where(
                torch.logical_and(freqs > cutoffs[0], freqs < cutoffs[1]), 1.0, 0.0
            ).float()
        elif filt_type == FilterType.BANDSTOP:
            freq_resp = torch.where(
                torch.logical_or(freqs < cutoffs[0], freqs > cutoffs[1]), 1.0, 0.0
            ).float()
        else:
            raise ValueError(f"Unrecognized filter type: {filt_type.value}")
        # create impulse response by windowing
        ir = torch.fft.irfft(freq_resp, n=filt_size, dim=-1)
        filter_window = torch.kaiser_window(filt_size, dtype=torch.float32).roll(
            filt_size // 2, -1
        )
        self.ir_windowed = (filter_window * ir)[None, None, :].to(
            self.ir_windowed.device
        )
        self.filt_type = filt_type
        self.cutoffs = cutoffs
        self.sample_rate = sample_rate
        self.filt_size = filt_size
        self.delay = filt_size // 2  # constant group delay

    def forward(
        self,
        audio: torch.Tensor,
    ):
        """Process audio with filter

        Args:
            audio (torch.Tensor): input audio [n_channels, n_samples]

        Returns:
            torch.Tensor: filtered audio
        """
        n_channels, orig_len = audio.shape
        # standard convolution implementation
        # pad input with cache
        audio = torch.cat([self.cache[:n_channels], audio], dim=-1)
        self.cache = audio[:, -(self.filt_size - 1) :]
        filtered = F.conv1d(
            audio[:, None, :],
            self.ir_windowed,
            padding="valid",
        ).squeeze(1)
        return filtered


class IIRFilter(nn.Module):
    def __init__(
        self,
        filt_type: FilterType,
        cutoff: float,
        resonance: float,
    ):
        """Time-invariant IIR filter

        Args:
            filt_type (FilterType): Type of the filter (FilterType.LOWPASS/HIGHPASS/BANDPASS).
            cutoff (float): Cutoff frequency in Hz (0 < cutoff < f_nyq)
            resonance (float): Filter resonance, controls bandwidth in case of bandpass
            sample_rate (int): Sampling rate
        """
        super().__init__()
        # register buffer only allowed once
        self.register_buffer("g", torch.empty(1, 1, 1))
        self.register_buffer("twoR", torch.empty(1, 1, 1) / resonance)
        self.register_buffer("mix", torch.empty(1, 1, 3))
        # Pass in fake sample rate for filter
        # Sample rate should be automatically overwritten by calling
        # set_parameters() from w2wbase.set_model_sample_rate_and_buffer_size()
        self.set_parameters(filt_type, cutoff, resonance, 48000)
        self.svf = _SVFLayer()

    def set_parameters(
        self,
        filt_type: Optional[FilterType] = None,
        cutoff: Optional[float] = None,
        resonance: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ):
        filt_type = self.filt_type if filt_type is None else filt_type
        cutoff = self.cutoff if cutoff is None else cutoff
        resonance = self.resonance if resonance is None else resonance
        sample_rate = self.sample_rate if sample_rate is None else sample_rate

        cutoff = max(min(cutoff, sample_rate / 2 - 1e-4), 1e-4)
        resonance = max(resonance, 1e-4)
        # frequency warping
        self.g = torch.ones(1, 1, 1, device=self.g.device) * math.tan(
            math.pi / sample_rate * cutoff
        )
        self.twoR = torch.ones(1, 1, 1, device=self.twoR.device) / resonance
        if filt_type == FilterType.LOWPASS:
            self.mix = torch.tensor([[[0.0, 1.0, 0.0]]], device=self.mix.device)
        elif filt_type == FilterType.HIGHPASS:
            self.mix = torch.tensor([[[0.0, 0.0, 1.0]]], device=self.mix.device)
        elif filt_type == FilterType.BANDPASS:
            self.mix = torch.tensor([[[1.0, 0.0, 0.0]]], device=self.mix.device)
        else:
            raise ValueError(f"Unrecognized filter type: {filt_type}")
        self.filt_type = filt_type
        self.cutoff = cutoff
        self.resonance = resonance
        self.sample_rate = sample_rate
        self.delay = 0

    def forward(self, audio: torch.Tensor):
        """pass through highpass filter

        Args:
            audio (torch.Tensor): [batch_size (or n_channels), n_samples]
        """
        batch_size, n_samples = audio.shape
        g = self.g.expand(n_samples, batch_size, -1)
        twoR = self.twoR.expand(n_samples, batch_size, -1)
        mix = self.mix.expand(n_samples, batch_size, -1)
        return self.svf(audio.permute(1, 0), g, twoR, mix)


class IIRSVF(nn.Module):
    def __init__(self):
        """
        Time-varying SVF with IIRs
        """
        super().__init__()
        self.svf = _SVFLayer()
        self.delay = 0

    def forward(
        self,
        audio: torch.Tensor,
        cutoff: torch.Tensor,
        resonance: torch.Tensor,
        mix: torch.Tensor,
        sample_rate: int,
    ):
        """Feed into time-varying svf

        Args:
            audio (torch.Tensor): Input audio [batch_size (or n_channels), n_samples]
            cutoff (torch.Tensor): Cutoff frequency [batch_size, n_samples, 1]
            resonance (torch.Tensor): Resonance (0 ~ 1), [batch_size, n_samples, 1]
            mix (torch.Tensor): Mix coeff. bp, lp and hp [batch_size, n_samples, 3] ex.) [[[1.0, 0.0, 0.0]]] = bandpass

        Returns:
            audio (torch.Tensor): [n_channels, n_samples]
        """
        cutoff = torch.clamp(cutoff, min=1e-4, max=sample_rate / 2 - 1e-4)
        resonance = torch.clamp(resonance, min=1e-4)
        g = torch.tan(math.pi / sample_rate * cutoff).permute(1, 0, 2)
        twoR = 1 / resonance.permute(1, 0, 2)
        mix = mix.permute(1, 0, 2)
        return self.svf(audio.permute(1, 0), g, twoR, mix)


class _SVFLayer(nn.Module):
    """
    SVF implementation based on "Time-varying filters for musical applications" [Wishnick, 2014]
    NOTE: This SVF is slow for use in training due to recurrent operations
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(1, 2))
        self.register_buffer("Y", torch.empty(4096, 2, 2))

    def forward(
        self,
        audio: torch.Tensor,
        g: torch.Tensor,
        twoR: torch.Tensor,
        mix: torch.Tensor,
    ):
        """pass audio through SVF
        Args:
            *** time-first, batch-second ***
            audio (torch.Tensor): [n_samples, batch_size]
            All filter parameters are [n_samples, batch_size, 1 or 3 (mix)]
            g (torch.Tensor): Normalized cutoff parameter
            twoR (torch.Tensor): Damping parameter
            mix (torch.Tensor): Mixing coefficient of bp, lp and hp

        Returns:
            [torch.Tensor]: Filtered audio. Shape [batch, n_samples]
        """
        seq_len, batch_size = audio.shape
        T = 1.0 / (1.0 + g * (g + twoR))
        H = T.unsqueeze(-1) * torch.cat(
            [torch.ones_like(g), -g, g, twoR * g + 1], dim=-1
        ).reshape(seq_len, batch_size, 2, 2)

        # Y = gHBx + Hs
        gHB = g * T * torch.cat([torch.ones_like(g), g], dim=-1)
        # [n_samples, batch_size, 2]
        gHBx = gHB * audio.unsqueeze(-1)
        if seq_len > self.Y.shape[0]:
            self.Y = torch.empty(seq_len, 2, 2, device=self.Y.device)
        Y = self.Y[:seq_len, :batch_size, :]
        # initialize filter state
        state = self.state.expand(batch_size, -1)
        for t in range(seq_len):
            Y[t] = gHBx[t] + torch.bmm(H[t], state.unsqueeze(-1)).squeeze(-1)
            state = 2 * Y[t] - state
        self.state = state

        # HP = x - 2R*BP - LP
        y_hps = audio - twoR.squeeze(-1) * Y[:, :, 0] - Y[:, :, 1]

        y_mixed = (
            twoR.squeeze(-1) * mix[:, :, 0] * Y[:, :, 0]
            + mix[:, :, 1] * Y[:, :, 1]
            + mix[:, :, 2] * y_hps
        )
        y_mixed = y_mixed.permute(1, 0)
        return y_mixed
