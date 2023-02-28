"""
Code based off https://github.com/csteinmetz1/steerable-nafx/blob/main/steerable-nafx.ipynb
"""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.tcn import TCN
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class OverdriveModel(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 n_blocks: int = 4,
                 channel_width: int = 32,
                 kernel_size: int = 13,
                 dilation_growth: int = 2,
                 n_params: int = 2,
                 cond_dim: int = 32) -> None:
        super().__init__()

        # MLP layers for conditioning
        self.n_controls = n_params
        self.control_to_cond_network = nn.Sequential(
            nn.Linear(n_params, cond_dim // 2),
            nn.ReLU(),
            nn.Linear(cond_dim // 2, cond_dim),
            nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.ReLU(),
        )

        # TCN model
        out_channels = [channel_width] * n_blocks
        dilations = [dilation_growth ** n for n in range(n_blocks)]
        self.tcn = TCN(out_channels,
                       dilations,
                       in_ch,
                       kernel_size,
                       cond_dim=cond_dim,
                       use_film_bn=True,
                       is_cached=True)
        self.output = nn.Conv1d(out_channels[-1], out_ch, kernel_size=(1,))

        # Random initialization
        self.init_weights()

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        cond = self.control_to_cond_network(params)  # Map params to conditioning vector
        x = self.tcn(x, cond)  # Process the dry audio
        x = self.output(x)  # Convert to 1 channel
        x = tr.tanh(x)  # Ensure the wet audio is between -1 and 1
        return x

    def init_weights(self,
                     linear_std: float = 0.4,
                     conv_std: float = 0.3,
                     output_std: Optional[float] = 0.25) -> None:
        for layer in self.control_to_cond_network:
            if layer.__class__.__name__ == 'Linear':
                nn.init.normal_(layer.weight, 0, linear_std)
        for block in self.tcn.blocks:
            nn.init.normal_(block.conv.conv.weight, 0, conv_std)
        if output_std is not None:
            nn.init.normal_(self.output.weight, 0, output_std)


class OverdriveModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "conv1d-overdrive.random"

    def get_model_authors(self) -> List[str]:
        return ["Nao Tokui"]

    def get_model_short_description(self) -> str:
        return "Neural distortion/overdrive effect"

    def get_model_long_description(self) -> str:
        return "Neural distortion/overdrive effect through randomly initialized Convolutional Neural Network"

    def get_technical_description(self) -> str:
        return "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "2.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2010.04237",
            "Code": "https://github.com/csteinmetz1/micro-tcn"
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2020). Randomized overdrive neural networks. arXiv preprint arXiv:2010.04237."

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [NeutoneParameter("depth", "Effect Depth", 0.5),
                NeutoneParameter("P1", "Feature modulation 1", 0.5),
                NeutoneParameter("P2", "Feature modulation 2", 0.5)]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # conditioning for FiLM layer
        p1 = params["P1"]
        p2 = params["P2"]
        depth = params["depth"]
        cond = tr.stack([p1, p2], dim=1) * depth
        cond = cond.expand(2, cond.size(1))
        x = x.unsqueeze(1)
        x = self.model(x, cond)
        x = x.squeeze(1)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = Path(args.output)

    model = OverdriveModel()
    wrapper = OverdriveModelWrapper(model)
    metadata = wrapper.to_metadata()
    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=True, submission=True
    )
