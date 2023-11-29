import logging
import os
from typing import Optional

import torch
from torch import Tensor
from torch import nn

from neutone_sdk.conv1dcausal import Conv1dCausal
from neutone_sdk.tcn_1d import FiLM

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class GatedAF(nn.Module):
    """Gated activation function
    applies a tanh activation to one half of the input
    and a sigmoid activation to the other half, and then multiplies them element-wise.

    Returns:
        Tensor: The output of the gated activation function.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        x_tanh, x_sigmoid = x.chunk(2, dim=1)  # Split the output into two halves

        x_tanh = torch.tanh(x_tanh)  # Apply tanh activation
        x_sigmoid = torch.sigmoid(x_sigmoid)  # Apply sigmoid activation

        # Element-wise multiplication of tanh and sigmoid activations
        x = x_tanh * x_sigmoid
        return x


class GCN1DBlock(nn.Module):
    """Single block of a Gated Convolutional Network (GCN) with conditional modulation.

    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        dilation (int, optional): Dilation rate for dilated convolutions.
        stride (int, optional): Stride for the convolution.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        cond_dim: int = 0,
        use_bias_in_conv: bool = False,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch * 2,  # adapt for the Gated Activation Function
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=use_bias_in_conv,
        )

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim=cond_dim, num_features=out_ch * 2)

        self.gated_activation = GatedAF()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x = self.conv(x)  # Apply causal convolution
        if (
            cond is not None and self.film is not None
        ):  # Apply FiLM if conditional input is given
            x = self.film(x, cond)
        # Apply gated activation function
        x = self.gated_activation(x)
        # Apply residual convolution and add to output
        x_res = self.res(x_in)
        x = x + x_res
        return x


class GCN1D(nn.Module):
    """Gated Convolutional Network (GCN) model, often used in sequence modeling tasks.

    Parameters:
        in_ch (int, optional): Number of input channels.
        out_ch (int, optional): Number of output channels.
        n_blocks (int, optional): Number of GCN blocks.
        n_channels (int, optional): Number of channels in the GCN blocks.
        dilation_growth (int, optional): Growth rate for dilation in the GCN blocks.
        kernel_size (int, optional): Size of the convolution kernel.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the GCN model.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 10,
        n_channels: int = 64,
        dil_growth: int = 4,
        kernel_size: int = 13,
        cond_dim: int = 0,
        use_act: bool = True,
        use_bias_in_conv: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dil_growth = dil_growth
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.use_act = use_act
        self.use_bias_in_conv = use_bias_in_conv

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_blocks
        self.dilations = [dil_growth**idx for idx in range(n_blocks)]

        # Blocks number is given by the number of elements in the channels list
        self.n_blocks = len(self.channels)
        assert len(self.dilations) == self.n_blocks

        # Create a list of strides
        self.strides = [1] * self.n_blocks

        # Create a list of GCN blocks
        self.blocks = nn.ModuleList()
        block_out_ch = None

        for idx, (curr_out_ch, dil, stride) in enumerate(
            zip(self.channels, self.dilations, self.strides)
        ):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            self.blocks.append(
                GCN1DBlock(
                    block_in_ch,
                    block_out_ch,
                    self.kernel_size,
                    dilation=dil,
                    stride=stride,
                    cond_dim=self.cond_dim,
                    use_bias_in_conv=self.use_bias_in_conv,
                )
            )

        # Output layer
        self.out_net = nn.Conv1d(
            self.channels[-1], out_ch, kernel_size=(1,), stride=(1,), bias=False
        )

        # Activation function
        self.act = nn.Tanh()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if cond is not None:
            assert cond.ndim == 2  # (batch_size, cond_dim)
        for block in self.blocks:  # Apply GCN blocks
            x = block(x, cond)
        x = self.out_net(x)  # Apply output layer

        if self.act is not None:
            x = self.act(x)  # Apply tanh activation function
        return x

    def calc_receptive_field(self) -> int:
        """Calculate the receptive field of the model.
        The receptive field is the number of input samples that affect the output of a block.

        The receptive field of the model is the sum of the receptive fields of all layers:
        RF = 1 + \sum_{i=1}^{n}(kernel\_size_i - 1) \cdot dilation_i

        i is the layer index, n is the number of layers.

        Returns:
            int: The receptive field of the model.
        """
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf
