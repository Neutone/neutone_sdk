"""
Based off
https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb
"""
import logging
import os
from typing import Optional, List, Dict
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import torch
from torch import Tensor
from torch import nn
import torchaudio
from torchaudio.transforms import Spectrogram
from neutone_sdk.stream_conv import (
    StreamConv1d,
    AlignBranches,
    get_same_pads,
    switch_streaming_mode,
)
from neutone_sdk.utils import save_neutone_model
from neutone_sdk.parameter import NeutoneParameter
from neutone_sdk.wavform_to_wavform import WaveformToWaveformBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class FiLM(nn.Module):
    def __init__(
        self,
        cond_dim: int,  # dim of conditioning input
        num_features: int,  # dim of the conv channel
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_features, affine=False)
        else:
            self.bn = nn.Identity()
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert cond.ndim == 2
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine
        return x


class TCN1DBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        padding: List[int] = None,
        cond_dim: int = 3,
        use_bias_in_conv: bool = True,
        use_bn: bool = True,
        use_act: bool = True,
        act: Optional[nn.Module] = None,
        prelu_ch: int = 1,
        res_groups: int = 1,
    ) -> None:
        super().__init__()
        self.padding = padding
        self.conv = StreamConv1d(
            in_ch,
            out_ch,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding,
        )
        assert cond_dim > 0
        self.film = FiLM(COND_DIM, out_ch, use_bn=use_bn)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_ch)
        else:
            self.bn = nn.Identity()
        if use_act:
            if act is None:
                self.act = nn.PReLU(prelu_ch)
        else:
            self.act = nn.Identity()
        self.res = nn.Conv1d(in_ch, out_ch, (1,), groups=res_groups, bias=False)
        self.align = AlignBranches(2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)
        x = self.film(x, cond)
        x = self.bn(x)
        x = self.act(x)
        outs = self.align([self.res(x_in), x])
        return outs[0] + outs[1]


class TCN1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 10,
        kernel_size: int = 13,
        n_channels: int = 64,
        dil_growth: int = 4,
        cond_dim: int = 3,
        use_act: bool = True,
        use_bn: bool = False,
        use_bias_in_conv: bool = True,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dil_growth = dil_growth
        self.n_blocks = n_blocks
        self.stack_size = n_blocks
        self.cond_dim = cond_dim
        self.use_act = use_act
        self.use_bn = use_bn
        self.use_bias_in_conv = use_bias_in_conv
        self.blocks = nn.ModuleList()
        pad_mode = "causal" if is_causal else "noncausal"
        for n in range(self.n_blocks):
            if n == 0:
                block_in_ch = in_ch
                block_out_ch = self.n_channels
            elif n == self.n_blocks - 1:
                block_in_ch = self.n_channels
                block_out_ch = out_ch
            else:
                block_in_ch = self.n_channels
                block_out_ch = self.n_channels

            dilation = self.dil_growth**n
            padding = get_same_pads(
                kernel_size, stride=1, dilation=dilation, mode=pad_mode
            )
            self.blocks.append(
                TCN1DBlock(
                    block_in_ch,
                    block_out_ch,
                    self.kernel_size,
                    dilation,
                    padding=padding,
                    cond_dim=self.cond_dim,
                    use_act=self.use_act,
                    use_bn=self.use_bn,
                    use_bias_in_conv=self.use_bias_in_conv,
                )
            )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        for block in self.blocks:
            x = block(x, cond)
        return x

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples."""
        rf = self.kernel_size
        for idx in range(1, self.n_blocks):
            dilation = self.dil_growth ** (idx % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf


class MultiSpecLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[64, 128, 256, 512, 1024, 2048],
        win_lengths=None,
        hop_lengths=None,
        mag_w=1.0,
        log_mag_w=1.0,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes
        win_lengths = fft_sizes if win_lengths is None else win_lengths
        if hop_lengths is None:
            overlap = 0.75
            hop_lengths = [int((1 - overlap) * s) for s in fft_sizes]
        self.specs = nn.ModuleList(
            [
                Spectrogram(n_fft, wl, hl, center=False)
                for n_fft, wl, hl in zip(fft_sizes, win_lengths, hop_lengths)
            ]
        )
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths

    def forward(self, input_audio, target_audio):
        loss = 0
        for spec in self.specs:
            x_spec = spec(input_audio)
            y_spec = spec(target_audio)
            loss += self.mag_w * torch.mean(torch.abs(x_spec - y_spec))
            loss += self.log_mag_w * torch.mean(
                torch.abs(torch.log(x_spec + 1e-6) - torch.log(y_spec + 1e-6))
            )
        return loss


class TCNModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "tcn.SOMETHING"

    def get_model_authors(self) -> List[str]:
        return ["YOUR NAME HERE"]

    def get_model_short_description(self) -> str:
        return "Neural SOMETHING effect"

    def get_model_long_description(self) -> str:
        return "Neural SOMETHING effect through Time Convolutional Neural Network"

    def get_technical_description(self) -> str:
        return "Random SOMETHING effect through Temporal 1D-convolution layers. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["type of effect"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2112.02926",
            "Code": "https://github.com/csteinmetz1/steerable-nafx",
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2021). Steerable discovery of neural audio effects. at 5th Workshop on Creativity and Design at NeurIPS."

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("depth", "Modulation Depth", 0.5),
            NeutoneParameter("P1", "Feature modulation 1", 0.0),
            NeutoneParameter("P2", "Feature modulation 2", 0.0),
            NeutoneParameter("P3", "Feature modulation 3", 0.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # conditioning for FiLM layer
        p1 = params["P1"]
        p2 = params["P2"]
        p3 = params["P3"]
        depth = params["depth"]
        cond = torch.stack([p1, p2, p3], dim=1) * depth
        cond = cond.expand(x.shape[0], 3)
        x = x.unsqueeze(1)
        x = self.model(x, cond)
        x = x.squeeze(1)
        return x


COND_DIM = 3

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dry", type=str, help="audio file with no FX")
    parser.add_argument("wet", type=str, help="same audio with FX applied")
    parser.add_argument("output", type=str, help="output folder")
    parser.add_argument("--sr", type=int, default=48000, help="sampling rate")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--n_iters", type=int, default=2500)
    parser.add_argument("--slice_len", type=int, default=50000)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # resample audio
    x, dry_sr = torchaudio.load(args.dry)
    x = torchaudio.functional.resample(x, dry_sr, args.sr)
    y, wet_sr = torchaudio.load(args.wet)
    y = torchaudio.functional.resample(y, wet_sr, args.sr)
    if not y.shape[-1] == x.shape[-1]:
        print(
            f"Input and output files are different lengths! Found clean: {x.shape[-1]} processed: {y.shape[-1]}."
        )
    if y.shape[-1] > x.shape[-1]:
        print(f"Cropping target...")
        y = y[:, : x.shape[-1]]
    else:
        print(f"Cropping input...")
        x = x[:, : y.shape[-1]]
    # make audio mono (use left channel) (batch, channel, time)
    proc_x = x[None, 0:1, :]
    proc_y = y[None, 0:1, :]
    # conditioning
    c = torch.zeros(1, COND_DIM, device=device, requires_grad=False)
    # build the model
    model = TCN1D(n_blocks=4, cond_dim=COND_DIM, use_bn=True, is_causal=args.causal)
    rf = model.calc_receptive_field()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count}, Receptive field: {rf}")

    # setup loss function, optimizer, and scheduler
    loss_fn = MultiSpecLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_lengths=[16, 64, 256, 1024],
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    ms1 = int(args.n_iters * 0.8)
    ms2 = int(args.n_iters * 0.95)
    milestones = [ms1, ms2]
    print(
        "Learning rate schedule:",
        f"1:{args.lr:0.2e} ->",
        f"{ms1}:{args.lr*0.1:0.2e} ->",
        f"{ms2}:{args.lr*0.01:0.2e}",
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=0.1,
        verbose=False,
    )

    # move model and tensors to GPU
    model = model.to(device)
    proc_x = proc_x.to(device)
    proc_y = proc_y.to(device)
    c = c.to(device)
    loss_fn = loss_fn.to(device)

    slice_len = min(x.shape[-1], args.slice_len)

    # iteratively update the weights
    pbar = tqdm(range(args.n_iters))
    losses = []
    for n in pbar:
        optimizer.zero_grad()
        # crop both input/output randomly
        start_idx = torch.randint(0, proc_x.shape[-1] - slice_len, (1,))[0]
        x_crop = proc_x[..., start_idx : start_idx + slice_len]
        y_crop = proc_y[..., start_idx : start_idx + slice_len]
        y_hat = model(x_crop, c)
        # crop output bc first rf samples don't have proper context
        loss = loss_fn(y_hat[..., rf:], y_crop[..., rf:])
        loss.backward()
        losses.append(loss.detach().cpu())
        optimizer.step()
        scheduler.step()
        if (n + 1) % 10 == 0:
            pbar.set_description(f" Loss: {loss.item():0.3e} | ")

    # export to neutone
    model.eval()
    switch_streaming_mode(model)
    model = torch.jit.script(model.to("cpu"))
    model(torch.zeros(1, 1, 4096), torch.zeros(1, 3))
    wrapped = TCNModelWrapper(model)
    save_neutone_model(
        wrapped,
        Path(args.output),
        freeze=False,
        dump_samples=True,
        submission=True,
    )
