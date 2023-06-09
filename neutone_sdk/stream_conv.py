import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math

EVAL_MAX_BATCH_SIZE = 8


def get_same_pads(kernel_size: int, stride: int = 1, dilation=1, mode="causal"):
    """Calculates 'same' padding. This results in output length being the same as input when stride == 1. Otherwise, the output length is input//stride.

    Args:
        kernel_size (int): kernel size of the convolution
        dilation (int, optional): Dilation of the convolution. Defaults to 1.
        mode (str, optional): Type of convolution padding ('causal' or 'noncausal'). 'causal' convolutions pad only the left side, which results in output at timestep t relying only on steps before time t resulting in a realtime model. In 'noncausal' padding, convolution kernel to calculate the output at timestep t is centered around timestep t; thus we need to know the input at timesteps t+1,...,t+k//2. This can add latency during streaming. Defaults to 'causal'.

    Returns:
        Tuple[int]: (left_padding, right_padding)
    """
    if kernel_size == 1:
        return (0, 0)
    K = (kernel_size - 1) * dilation + 1
    pad_total = K - stride
    if mode == "causal":
        return (pad_total, 0)
    elif mode == "noncausal":
        return (pad_total - pad_total // 2, pad_total // 2)
    else:
        raise ValueError(
            f"Convolution padding mode: {mode} not recognized. Only 'causal' and 'noncausal' are supported"
        )


class StreamConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Tuple[int] = (0, 0),
        dilation: int = 1,
        extra_pad: bool = False,
        **kwargs,
    ):
        """
        Streaming (cached) 1d convolution. Parameters are all the same as nn.Conv1d except padding being a tuple (left and right padding can be specified).
        Simply adding zero padding results in discontinuities between buffers.
        Streaming mode keeps a cache of past inputs to keep the results same as if the buffers were concatenated.
        """
        self.pads = padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,  # don't use internal padding
            dilation=dilation,
            **kwargs,
        )
        self.K = (
            dilation * (self.kernel_size[0] - 1) + 1
        )  # total space that a kernel occupies
        self.extra_pad = extra_pad  # fill last window during training
        self.register_buffer(
            "cache", torch.zeros(EVAL_MAX_BATCH_SIZE, in_channels, self.pads[0])
        )  # initialized with zero padding

    def get_n_frames(self, input_length: int) -> float:
        # data with size L allows for (L-K)//s conv ops
        return (input_length - self.K) / self.stride[0]

    def cached_pad(self, x: torch.Tensor) -> torch.Tensor:
        """
        self.cache = x[..., -padding:] doesn't work for non-"same" type convs
        This keeps track of where the convolution was performed last
        so that we can keep necessary samples to do next convolution with next buffer
        """
        # x: batch, channel, L
        x = torch.cat([self.cache[: x.shape[0]], x], dim=-1)
        n_convs = math.floor(self.get_n_frames(x.shape[-1]))
        # starting position of conv that wasn't calculated
        next_pos = self.stride[0] * (n_convs + 1)
        if next_pos < x.shape[-1]:
            # save as new cache
            self.cache = x[..., next_pos:].detach()
        else:
            # There is nothing that can be cached: shouldn't happen unless stride is larger than n
            self.cache = torch.empty(
                x.shape[0], self.in_channels, 0, device=self.cache.device
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = self.cached_pad(x)
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=0,  # no padding
                dilation=self.dilation,
                groups=self.groups,
            )
        else:  # during training use normal padding
            if self.extra_pad:
                # Sometimes, last window isn't filled and end of signal is wasted
                # This adds extra padding to end to prevent that
                # based on encodec.modules.conv.get_extra_padding_for_conv1d
                total_p = self.pads[0] + self.pads[1]
                L = x.shape[-1] + total_p
                n_frames = math.ceil(self.get_n_frames(L))
                ideal_length = n_frames * self.stride[0] + (self.K - total_p)
                extra_p = ideal_length - L
                x = F.pad(x, pad=(self.pads[0], self.pads[1] + extra_p))
            else:
                x = F.pad(x, pad=(self.pads[0], self.pads[1]))
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=0,  # no padding
                dilation=self.dilation,
                groups=self.groups,
            )

    @torch.jit.export
    def flush(self) -> torch.Tensor:
        """Flush remaining cache along with end padding
        Not necessary for streaming uses

        Returns:
            torch.Tensor: flushed output or empty tensor
        """
        x = F.pad(self.cache, pad=(0, self.pads[1]))
        if x.shape[-1] - self.K > 0:
            # there is enough to calculate
            return nn.functional.conv1d(
                x,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=0,  # no padding
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            return torch.empty(
                self.cache.shape[0], self.out_channels, 0, device=self.cache.device
            )


class StreamConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Tuple[int] = (0, 0),
        dilation: int = 1,
        **kwargs,
    ):
        """
        Streaming (cached) 1d transposed convolution. Parameters are all the same as nn.ConvTranspose1d except padding being a tuple (left and right padding can be specified).
        Padding for transposed conv equates to cropping of the output signal.
        Streaming mode accounts for outputs from last buffer overlapping into current output.
        """
        self.pads = padding
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,  # don't use internal padding
            dilation=dilation,
            **kwargs,
        )
        self.register_buffer(
            "cache", torch.zeros(EVAL_MAX_BATCH_SIZE, out_channels, self.pads[0])
        )
        self.use_bias: bool = kwargs.get("bias", True)
        # self.bias is Optional[Tensor] and bad for scripting
        self._bias: torch.Tensor = self.bias if self.use_bias else torch.empty(0)
        self.init: bool = True
        self.batch_size: int = EVAL_MAX_BATCH_SIZE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            next_pos = self.stride[0] * x.shape[-1]  # head of next conv
            self.batch_size = x.shape[0]
            out = nn.functional.conv_transpose1d(
                x,
                self.weight,
                None,  # bias added later
                stride=self.stride,
                padding=0,  # no padding (cropping)
                output_padding=self.output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )
            # add overlapping cache
            if self.cache.shape[-1] > out.shape[-1]:
                out = F.pad(out, pad=(0, self.cache.shape[-1] - out.shape[-1]))
            out[..., : self.cache.shape[-1]] += self.cache[: self.batch_size]
            if next_pos < out.shape[-1]:
                # save as new cache
                self.cache = out[..., next_pos:].detach()
                # crop output
                out = out[..., :next_pos]
            else:
                # There is nothing that can be cached
                self.cache = torch.empty(
                    out.shape[0], self.out_channels, 0, device=self.cache.device
                )
            if self.use_bias:
                out = out + self._bias[None, :, None]  # add bias after caching
            if self.init:  # apply left padding (cropping)
                out = out[..., self.pads[0] :]
                self.init = False
        else:  # during training, use normal padding (cropping)
            out = nn.functional.conv_transpose1d(
                x,
                self.weight,
                self._bias,
                stride=self.stride,
                padding=0,  # no padding (cropping)
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
            out = out[..., self.pads[0] : out.shape[-1] - self.pads[1]]
        return out

    @torch.jit.export
    def flush(self) -> torch.Tensor:
        out = self.cache[: self.batch_size, :, : self.cache.shape[-1] - self.pads[1]]
        if self.use_bias:
            return out + self._bias[None, :, None]
        else:
            return out


class AlignBranches(nn.Module):
    """
    Keeps cache of branches that are ahead
    Useful for residual nets with irregular pad sizes for the conv branch,
    where the output size fluctuates from the input size.
    Ex.) First output of non causal padding is shorter
    """

    def __init__(self, *branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.cached = [False for i in range(len(branches))]
        self.caches = [torch.empty(0) for i in range(len(branches))]
        self.has_flush: Tuple[bool] = [hasattr(b, "flush") for b in branches]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        ys = [branch(x) for branch in self.branches]
        lengths = [y.shape[-1] for y in ys]
        l_min = min(lengths)
        for i, y in enumerate(ys):
            if self.cached[i]:
                y_cache_len = l_min - self.caches[i].shape[-1]
                if y_cache_len > 0:  # output uses cache and y
                    out = y[..., :y_cache_len]
                    outs.append(torch.cat([self.caches[i], out], dim=-1))
                    self.caches[i] = y[..., y_cache_len:]
                    self.cached[i] = True
                else:  # output is covered by cache
                    out = self.caches[i][..., :l_min]
                    outs.append(out)
                    # remove used cache and add y
                    self.caches[i] = torch.cat([self.caches[i][..., l_min:], y], dim=-1)
            else:
                out = y[..., :l_min]
                self.caches[i] = y[..., l_min:]
                outs.append(out)
            self.cached[i] = True
        return outs

    # @torch.jit.export # flush currently doesn't work with torchscript
    def flush(self) -> List[torch.Tensor]:
        outs = []
        for i, b in enumerate(self.branches):
            if self.has_flush[i]:
                # print(self.has_flush[i], b, "hasflush")
                outs.append(torch.cat([self.caches[i], b.flush()], dim=-1))
            else:
                # print(self.has_flush[i], b, "noflush")
                outs.append(torch.cat([self.caches[i]], dim=-1))
        return outs
