import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class FIRFilter(nn.Module):
    def __init__(
        self,
        cutoffs: List[float],
        sample_rate: int,
        filt_size: int = 257,
        n_channels: int = 1,
        filt_type: str = "bandpass",
    ):
        """Streamable FIR filter for pre-filtering of model inputs, etc.

        Args:
            cutoffs (List[float]): Cutoff frequencies (in Hz). 2 should be given if bandpass/stop
            sample_rate (int): Sampling rate
            filt_size (int, optional): Length of the FIR. Defaults to 257.
            n_channels (int, optional): Number of channels of the input/output. Defaults to 1.
            filt_type (str, optional): Type of the filter (low/high/bandpass, bandstop). Defaults to "bandpass".
        """
        super().__init__()
        if len(cutoffs) == 2:
            if filt_type in ["highpass", "lowpass"]:
                raise ValueError("only 1 cutoff value supported for this filter type")
        else:
            if filt_type in ["bandpass", "bandstop"]:
                raise ValueError("2 cutoff values (low, high) needed for this type")
        # create frequency response by frequency sampling
        freqs = torch.fft.rfftfreq(filt_size, 1 / sample_rate)

        if filt_type == "highpass":
            freq_resp = torch.where((freqs > cutoffs[0]), 1.0, 0.0).float()
        elif filt_type == "lowpass":
            freq_resp = torch.where((freqs < cutoffs[0]), 1.0, 0.0).float()
        elif filt_type == "bandpass":
            freq_resp = torch.where(
                torch.logical_and(freqs > cutoffs[0], freqs < cutoffs[1]), 1.0, 0.0
            ).float()
        elif filt_type == "bandstop":
            freq_resp = torch.where(
                torch.logical_or(freqs < cutoffs[0], freqs > cutoffs[1]), 1.0, 0.0
            ).float()
        else:
            raise ValueError(f"Unrecognized filter type {filt_type}")

        # create impulse response by windowing
        ir = torch.fft.irfft(freq_resp, n=filt_size, dim=-1)
        filter_window = torch.kaiser_window(filt_size, dtype=torch.float32).roll(
            filt_size // 2, -1
        )
        ir_windowed = filter_window * ir
        self.register_buffer("ir_windowed", ir_windowed[None, :])
        self.filt_size = filt_size
        self.sample_rate = sample_rate
        self.register_buffer("cache", torch.zeros(n_channels, filt_size - 1))

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
        audio = torch.cat([self.cache, audio], dim=-1)
        self.cache = audio[..., -(self.filt_size - 1) :]
        filtered = F.conv1d(
            audio.unsqueeze(0),
            self.ir_windowed[:, None, :].expand(-1, n_channels, -1),
            padding="valid",
        ).squeeze(0)
        return filtered


class IIRFilter(nn.Module):
    def __init__(
        self,
        cutoff: float,
        resonance: float,
        sample_rate: int,
        filt_type: str = "lowpass",
    ):
        """Time-invariant IIR filter

        Args:
            cutoff (float): cutoff frequency in Hz (0 < cutoff < f_nyq)
            resonance (float): filter resonance
            sample_rate (int): sampling rate
            filt_type (int): filter type ('lowpass', 'highpass', 'bandpass')
        """
        super().__init__()
        cutoff = max(min(cutoff, sample_rate / 2 - 1e-4), 0)
        resonance = max(resonance, 1e-4)
        self.register_buffer(
            # frequency warping
            "g",
            torch.ones(1, 1, 1) * math.tan(math.pi / sample_rate * cutoff),
        )
        self.register_buffer("twoR", torch.ones(1, 1, 1) / resonance)
        if filt_type == "lowpass":
            self.register_buffer("mix", torch.tensor([[[0.0, 1.0, 0.0]]]))
        elif filt_type == "highpass":
            self.register_buffer("mix", torch.tensor([[[0.0, 0.0, 1.0]]]))
        elif filt_type == "bandpass":
            self.register_buffer("mix", torch.tensor([[[1.0, 0.0, 0.0]]]))
        self.svf = torch.jit.script(_SVFLayer())

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
        """time-varying SVF with IIRs"""
        super().__init__()
        self.svf = torch.jit.script(_SVFLayer())

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
            audio (torch.Tensor): input audio [batch_size (or n_channels), n_samples]
            cutoff (torch.Tensor): Cutoff frequency [batch_size, n_samples, 1]
            resonance (torch.Tensor): resonance (0 ~ 1) [batch_size, n_samples, 1]
            mix (torch.Tensor): Mix coeff. bp, lp and hp [batch_size, n_samples, 3] ex.) [[[1.0, 0.0, 0.0]]] = bandpass

        Returns:
            audio (torch.Tensor): [batch_size (or n_channels), n_samples]
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
            All filter parameters are [n_samples, batch_size, 1or3]
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

        Y = torch.empty(seq_len, batch_size, 2, device=audio.device)
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
        y_mixed = y_mixed.permute(1, 0).contiguous()
        return y_mixed
