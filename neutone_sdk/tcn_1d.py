"""
Based off
https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb
"""
import logging
import os
from typing import Optional

import torch as tr
from torch import Tensor
from torch import nn

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def causal_crop(x: Tensor, length: int) -> Tensor:
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x


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
        self.adaptor = nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert cond.ndim == 2
        cond = self.adaptor(cond)
        g, b = tr.chunk(cond, 2, dim=-1)
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)

        if self.use_bn:
            x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine

        return x


class TCN1DBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int,
                 dilation: int,
                 padding: Optional[int] = None,
                 cond_dim: int = 0,
                 use_bias_in_conv: bool = True,
                 use_bn: bool = True,
                 use_act: bool = True,
                 use_res: bool = True,
                 act: Optional[nn.Module] = None,
                 prelu_ch: int = 1,
                 res_groups: int = 1) -> None:
        super().__init__()
        self.padding = padding
        if self.padding is None:
            self.padding = ((kernel_size - 1) // 2) * dilation
        if act is None:
            act = nn.PReLU(prelu_ch)

        self.act = None
        if use_act:
            self.act = act

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            (kernel_size,),
            dilation=(dilation,),
            padding=self.padding,
            bias=use_bias_in_conv,
        )

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_ch, use_bn=use_bn)

        self.bn = None
        if use_bn and self.film is None:
            self.bn = nn.BatchNorm1d(out_ch)

        self.res = None
        if use_res:
            self.res = nn.Conv1d(in_ch,
                                 out_ch,
                                 (1,),
                                 groups=res_groups,
                                 bias=False)

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        x_in = x
        x = self.conv(x)
        if cond is not None and self.film is not None:
            x = self.film(x, cond)
        elif self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        if self.res is not None:
            res = self.res(x_in)
            x_res = causal_crop(res, x.shape[-1])
            x += x_res

        return x


class TCN1D(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_blocks: int = 10,
                 kernel_size: int = 13,
                 n_channels: int = 64,
                 dil_growth: int = 4,
                 padding: Optional[int] = None,
                 cond_dim: int = 0,
                 use_act: bool = True,
                 use_bn: bool = False,
                 use_bias_in_conv: bool = True) -> None:
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

            dilation = self.dil_growth ** n
            self.blocks.append(TCN1DBlock(
                block_in_ch,
                block_out_ch,
                self.kernel_size,
                dilation,
                padding=padding,
                cond_dim=self.cond_dim,
                use_act=self.use_act,
                use_bn=self.use_bn,
                use_bias_in_conv=self.use_bias_in_conv,
            ))

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if cond is not None:
            assert cond.ndim == 2  # (batch_size, cond_dim)
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


if __name__ == '__main__':
    tcn = TCN1D(n_blocks=4, cond_dim=3, use_bn=True)
    log.info(tcn.calc_receptive_field())
    audio = tr.rand((1, 1, 65536))
    cond = tr.rand((1, 3))
    # cond = None
    out = tcn.forward(audio, cond)
    log.info(out.shape)
