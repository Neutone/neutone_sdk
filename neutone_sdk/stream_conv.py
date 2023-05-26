import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

MAX_BATCH_SIZE = 8


def calculate_pads(kernel_size, dilation=1, mode="causal"):
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
    p = (kernel_size - 1) * dilation + 1
    half_p = p // 2
    if mode == "causal":
        return (2 * half_p, 0)
    elif mode == "noncausal":
        return (half_p, half_p)
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
        **kwargs,
    ):
        """Streaming (cached) 1d convolution. Parameters are all the same as nn.Conv1d except padding being a tuple (left and right padding can be specified)."""
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
        self.n = dilation * (self.kernel_size[0] - 1) + 1
        self.register_buffer(
            "cache", torch.zeros(MAX_BATCH_SIZE, in_channels, self.pads[0])
        )  # initialized with zero padding

    def cached_pad(self, x: torch.Tensor):
        # x: batch, channel, L
        x = torch.cat([self.cache[: x.shape[0]], x], dim=-1)
        # self.cache = x[..., -padding:] doesn't work for non-"same" type convs
        # data with size L allows for (L-n)//s conv ops
        n_convs = (x.shape[-1] - self.n) // self.stride[0]
        # starting position of conv that wasn't calculated
        next_pos = self.stride[0] * (n_convs + 1)
        if next_pos < x.shape[-1]:
            # save as new cache
            self.cache = x[..., next_pos:]
        else:
            # There is nothing that can be cached: shouldn't happen unless stride is larger than n
            self.cache = torch.empty(
                x.shape[0], self.in_channels, 0, device=self.cache.device
            )
        return x

    def forward(self, x):
        if not self.training:
            x = self.cached_pad(x)
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,  # no padding
                self.dilation,
                self.groups,
            )
        else:
            x = F.pad(x, pad=(self.pads[0], self.pads[1]))
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,  # no padding
                self.dilation,
                self.groups,
            )

    @torch.jit.export
    def flush(self):
        x = F.pad(self.cache, pad=(0, self.pads[1]))
        if x.shape[-1] - self.n > 0:
            # there is enough to calculate
            return nn.functional.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                0,  # no padding
                self.dilation,
                self.groups,
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
        """Streaming (cached) 1d transposed convolution. Parameters are all the same as nn.ConvTranspose1d except padding being a tuple (left and right padding can be specified)."""
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
            "cache", torch.zeros(MAX_BATCH_SIZE, out_channels, self.pads[0])
        )
        self.use_bias = kwargs.get("bias", True)
        # self.bias is Optional[Tensor]
        self._bias: torch.Tensor = self.bias if self.use_bias else torch.empty(0)
        self.init = True
        self.batch_size = MAX_BATCH_SIZE

    def forward(self, x: torch.Tensor):
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
            out[..., : self.cache.shape[-1]] += self.cache[: self.batch_size]
            if next_pos < out.shape[-1]:
                # save as new cache
                self.cache = out[..., next_pos:]
                # crop output
                out = out[..., :next_pos]
            else:
                # There is nothing that can be cached
                self.cache = torch.empty(
                    out.shape[0], self.out_channels, 0, device=self.cache.device
                )
            if self.use_bias:
                out = out + self._bias  # add bias after caching
            if self.init:  # apply left padding (cropping)
                out = out[..., self.pads[0] :]
                self.init = False
        else:
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
    def flush(self):
        out = self.cache[: self.batch_size, :, : self.cache.shape[-1] - self.pads[1]]
        if self.use_bias:
            return out + self._bias
        else:
            return out


class AlignBranches(nn.Module):
    # keep cache of branches that are ahead
    def __init__(self, *branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.cached = [False for i in range(len(branches))]
        self.caches = [torch.empty(0) for i in range(len(branches))]
        self.has_flush: Tuple[bool] = [hasattr(b, "flush") for b in branches]

    def forward(self, x: torch.Tensor):
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
    def flush(self):
        outs = []
        for i, b in enumerate(self.branches):
            if self.has_flush[i]:
                print(self.has_flush[i], b, "hasflushg")
                outs.append(torch.cat([self.caches[i], b.flush()], dim=-1))
            else:
                print(self.has_flush[i], b, "noflush")
                outs.append(torch.cat([self.caches[i]], dim=-1))
        return outs
