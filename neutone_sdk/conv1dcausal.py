import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class Conv1dCausal(nn.Module):
    """Causal 1D convolutional layer
    ensures outputs depend only on current and past inputs.

    Parameters:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        dilation (int, optional): Spacing between kernel elements.
        bias (bool, optional): If True, adds a learnable bias to the output.

    Returns:
        Tensor: The output of the causal 1D convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.padding = (
            kernel_size - 1
        ) * dilation  # input_len == output_len when stride=1
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            (kernel_size,),
            (stride,),
            padding=0,
            dilation=(dilation,),
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.padding, 0))  # standard zero padding
        x = self.conv(x)
        return x