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


class PaddingCached(nn.Module):
    """Maintains signal continuity over sample windows by caching the last `padding` samples.

    Attributes:
        padding (int): Number of samples to cache.
        channels (int): Number of channels in the input signal.
        pad (Tensor): Cached input signal.

    Returns:
        Tensor: Padded output.
    """

    def __init__(self, padding: int, channels: int) -> None:
        super().__init__()
        self.padding = padding
        self.channels = channels
        pad = torch.zeros(1, self.channels, self.padding)
        self.register_buffer("pad", pad)

    def forward(self, x: Tensor) -> Tensor:
        padded_x = torch.cat([self.pad, x], -1)     # concat input signal to the cache
        self.pad = padded_x[..., -self.padding :]   # discard old cache
        return padded_x


class Conv1dCached(nn.Module):
    """Conv1d with cache"""

    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        weight_norm: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.pad = PaddingCached(padding * 2, in_chan)
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel, stride, dilation=dilation, bias=bias
        )
        nn.init.normal_(self.conv.weight)  # random initialization
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        return x


class Conv1dSwitching(nn.Module):
    """Automatically switches between Conv1d, Conv1dCausal and Conv1dCached based on the model's mode.
    Should be able to handle 3 cases:

    1. Training/Evaluation phase: Conv1d
    2. Training/Evaluation phase: Conv1dCausal
    3. Inference phase: Conv1dCached

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride value for the convolution.
        padding (int, optional): Padding value. Defaults to 0.
        padding_mode (str, optional): Padding mode. Defaults to "zeros".
        dilation (int, optional): Dilation value. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        causal (bool, optional): Whether to use Conv1dCausal during training/evaluation. Defaults to True.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        padding_mode: str = "zeros",
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()

        self.causal = causal
        self.cached = False

        if self.causal:
            # For Training/Evaluation with Conv1dCausal
            self.active_conv_layer = Conv1dCausal(
                in_channels, out_channels, kernel_size, stride, dilation, bias
            )
        else:
            # For Training/Evaluation with nn.Conv1d
            self.active_conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                bias=bias,
            )

        # For Inference with Conv1dCached
        self.conv_cached = Conv1dCached(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

    # def prepare_for_inference(self) -> None:
    #     """Switch to Conv1dCached when preparing for inference."""
    #     print("Switching to Conv1dCached for inference...")
    #     self.cached = True

    # def train(self, mode: bool = True) -> None:
    #     """Switch to Conv1d or Conv1dCausal during training."""
    #     super().train(mode)
    #     self.cached = False

    # def eval(self) -> None:
    #     """Switch to Conv1d or Conv1dCausal during evaluation."""
    #     super().eval()

    def prepare_for_inference(self) -> None:
        print("Switching to Conv1dCached for inference...")
        self.cached = True
        self.active_conv_layer.eval()

    def forward(self, x: Tensor) -> Tensor:
        # Temporarily switch to the appropriate layer based on the current mode
        if self.cached == True:
            current_layer = self.conv_cached
        else:
            current_layer = self.active_conv_layer

        return current_layer(x)

if __name__ == "__main__":
    import torch
    from torchinfo import summary

    print("\n\nTest Conv1dSwitching with Conv1d")
    audio = torch.randn(1, 1, 65536)
    model = Conv1dSwitching(1, 1, 3, 1, 1, causal=False)
    model.eval()
    summary(model, input_data=audio)
    out = model(audio)
    print(out.shape)

    print("\n\nTest Conv1dSwitching with Conv1dCausal")
    audio = torch.randn(1, 1, 65536)
    model = Conv1dSwitching(1, 1, 3, 1, 1, causal=True)
    model.eval()
    summary(model, input_data=audio)
    out = model(audio)
    print(out.shape)

    print("\n\nTest Conv1dSwitching with Conv1dCached")
    audio = torch.randn(1, 1, 65536)
    model = Conv1dSwitching(1, 1, 3, 1, 1, causal=True)
    model.prepare_for_inference()
    summary(model, input_data=audio, depth=5, verbose=2)
    out = model(audio)
    print(out.shape)

    # TODO(francescopapaleo): make the layer scriptable
    # script_model = torch.jit.script(model)
    # script_model.save("conv1d_switching.pt")