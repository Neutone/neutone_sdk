import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import librosa as lbr
from pathlib import Path
from torchinfo import summary

# MODEL INFO
metadata = {
    "model_name": "conv-overdrive.random",
    "model_authors": ["Nao Tokui"],
    "model_short_description": "Neural Distortion/Overdrive effect",
    "model_long_description": "Neural Distortion/Overdrive effect through randomly initialized Convolutional Neural Network",
    "technical_description": "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. You'll get different types of distortion by re-initializing the weight or changing the activation function. Based on the idea proposed by Steinmetz et al.",
    "technical_links": {
        "Paper": "https://arxiv.org/abs/2010.04237",
        "Code": "https://csteinmetz1.github.io/ronn/",
        "Personal": "Christian J. Steinmetz, Joshua D. Reiss",
    },
    "tags": ["overdrive", "temporal convolution"],
    "model_type": "stereo-stereo",
    "sample_rate": 48000,
    "minimum_buffer_size": 32,
    "parameters": {
        "p1": {"used": "false", "name": "", "description": "", "type": "knob"},
        "p2": {"used": "false", "name": "", "description": "", "type": "knob"},
        "p3": {"used": "false", "name": "", "description": "", "type": "knob"},
        "p4": {"used": "false", "name": "", "description": "", "type": "knob"},
    },
    "version": 1,
    # do we need these??
    "domain_tags": ["music"],
    "short_description": "",
    "long_description": "",
    "labels": ["overdrive", "distortion"],
    "effect_type": "waveform-to-waveform",
    "multichannel": True,
}

# SETTINGS
configs = {
    "kernel_size": 13,
    "dilation_growth": 2,
    "nblocks": 4,
    "channel_growth": 0,
    "channel_width": 32,
    "optimizer": "adam",
    "ncondition": 2,
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CachedPadding(nn.Module):  # to maintain signal continuity over sample windows
    def __init__(self, padding, channels, pad_mode="constant"):
        super().__init__()
        self.padding = padding
        self.pad_mode = pad_mode
        self.channels = channels

        left_pad = torch.zeros(1, self.channels, self.padding)
        self.register_buffer("left_pad", left_pad)

    def forward(self, x):
        padded_x = torch.cat([self.left_pad, x], -1)
        self.left_pad = padded_x[..., -self.padding :]
        return padded_x


class CachedConv1d(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel,
        stride,
        padding,
        dilation=(1,),
        pad_mode="constant",
        weight_norm=False,
        bias=False,
    ):
        super().__init__()
        self.pad = CachedPadding(2 * padding, in_chan, pad_mode)
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel, stride, dilation=dilation, bias=bias
        )
        nn.init.normal_(self.conv.weight)  # random initialization
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        padding = kernel_size // 2 * dilation
        self.conv1 = CachedConv1d(
            in_ch,
            out_ch,
            kernel=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.film = FiLM(out_ch, 32)
        self.relu = nn.PReLU(out_ch)

    def forward(self, x, p):
        x = self.conv1(x)
        x = self.film(x, p)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FiLM(nn.Module):
    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)  # divide input into 2 chunks
        g = g.permute(0, 2, 1)  #
        b = b.permute(0, 2, 1)  #

        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class EffectModel(pl.LightningModule):
    def __init__(
        self,
        ninputs=1,
        noutputs=1,
        nblocks=4,
        channel_growth=2,
        channel_width=0,
        kernel_size=3,
        dilation_growth=2,
        ncondition=2,
    ):
        super().__init__()

        # MLP layers for conditioning
        self.ncondition = ncondition
        self.condition = torch.nn.Sequential(
            torch.nn.Linear(ncondition, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )

        # main model
        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width
            dilation = dilation_growth**n
            self.blocks.append(
                TCNBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
            )
        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)
        nn.init.normal_(self.output.weight)  # random initialization

    def forward(self, x, c):
        p = self.condition(c)  # conditioning

        for _, block in enumerate(self.blocks):
            x = block(x, p)
        y = torch.tanh(self.output(x))  # clipping

        return y


from auditioner_sdk.utils import (
    save_model,
    validate_metadata,
    get_example_inputs,
    test_run,
)
from auditioner_sdk import WaveformToWaveformBase


class OverdriveModelWrapper(WaveformToWaveformBase):
    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.condition = torch.tensor([0, 0], dtype=torch.float).reshape((1, 1, -1))

    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:

        # do any preprocessing here!
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        output = []
        for ch in range(x.shape[0]):  # process channel by channel
            x = x[ch].reshape(1, 1, -1)
            y = self.model(x, self.condition)
            output.append(y)
        output = torch.vstack(output).reshape(x.shape[0], -1).type_as(x)

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
        return output


if __name__ == "__main__":

    torch.set_grad_enabled(False)

    # create a root dir for our model
    root = Path("exports/overdrive_random")
    root.mkdir(exist_ok=True, parents=True)

    # get our model
    configs = dotdict(configs)
    model = EffectModel(
        ninputs=1,
        noutputs=1,
        nblocks=configs.nblocks,
        channel_growth=configs.channel_growth,
        channel_width=configs.channel_width,
        kernel_size=configs.kernel_size,
        dilation_growth=configs.dilation_growth,
        ncondition=configs.ncondition,
    )

    # model.to(torch.float)
    model.eval()
    model.freeze()
    # print(model)

    #    torchsummary.summary(model, [(1,65536), (1,2)], device="cpu")
    #    summary(model, input_size=[(1, 1, 1024), (1, 1, 2)])
    x = torch.randn((1, 1, 1024))
    c = torch.zeros((1, 1, 2))
    print(x)
    y = model.forward(x, c)
    print(y)

    # wrap it
    wrapper = OverdriveModelWrapper(model)

    # serialize it using torch.jit.script, torch.jit.trace,
    # or a combination of both.

    # option 1: torch.jit.script
    # using torch.jit.script is preferred for most cases,
    # but may require changing a lot of source code
    serialized_model = torch.jit.script(wrapper)

    # option 2: torch.jit.trace
    # using torch.jit.trace is typically easier, but you
    # need to be extra careful that your serialized model behaves
    # properly after tracing
    # example_inputs = get_example_inputs()
    # serialized_model = torch.jit.trace(wrapper, example_inputs[0],
    #                                     check_inputs=example_inputs)

    # take your model for a test run!
    test_run(serialized_model)

    # check that we created our metadata correctly
    success, msg = validate_metadata(metadata)
    assert success

    # save!
    save_model(serialized_model, metadata, root)
