"""
Code based off https://github.com/csteinmetz1/ronn
"""
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

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
                 activation: str = "ReLU",
                 init: str = "normal",
                 in_ch: int = 1,
                 n_blocks: int = 1,
                 channel_width: int = 1,
                 kernel_size: int = 3,
                 dilation_growth: int = 2,
                 n_params: int = 2,
                 cond_dim: int = 128) -> None:
        super().__init__()

        # MLP layers for conditioning
        self.n_controls = n_params
        self.control_to_cond_network = nn.Sequential(
            nn.Linear(n_params, cond_dim // 2),
            # nn.ReLU(),
            nn.Linear(cond_dim // 2, cond_dim),
            # nn.ReLU(),
            nn.Linear(cond_dim, cond_dim),
            # nn.ReLU(),
        )

        # TCN model
        out_channels = [channel_width] * n_blocks
        dilations = [dilation_growth ** n for n in range(n_blocks)]
        self.tcn = TCN(out_channels,
                       dilations,
                       in_ch,
                       kernel_size,
                       use_act=False,
                       use_res=False,
                       cond_dim=cond_dim,
                       use_film_bn=False,
                       is_cached=True)

        # Weight initialization
        self.init_weights(init)

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        print(f"in x {x.min()}")
        print(f"in x {x.max()}")
        cond = self.control_to_cond_network(params)  # Map params to conditioning vector
        x = self.tcn(x, cond)  # Process the dry audio
        # x = self.tcn(x)  # Process the dry audio
        # x = self.output(x)  # Convert to 1 channel
        # x = tr.tanh(x)  # Ensure the wet audio is between -1 and 1
        print(x.min())
        print(x.mean())
        print(x.max())
        return x

    def init_weights(self, init: str) -> None:
        for k, param in dict(self.named_parameters()).items():
            if "weight" in k:
                self.init_param_weight(param, init)

    @staticmethod
    def init_param_weight(param: Tensor, init: str) -> None:
        if init == "normal":
            nn.init.normal_(param, std=1)  # smooth
        elif init == "uniform":
            nn.init.uniform_(param, a=-0.1, b=0.1)  # harsh
        elif init == "dirac":
            nn.init.dirac_(param)  # nice, but only left channel
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(param)  # nice and smooth, even roomy
        elif init == "xavier_normal":
            nn.init.xavier_normal_(param)  # similar to uniform, harsher
        elif init == "kaiming_uniform":
            nn.init.kaiming_uniform_(param)  # hmm could be nice
        elif init == "orthongonal":
            nn.init.orthogonal_(param)  # inconsistent results
        else:
            raise ValueError(f"Invalid init: {init}")


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
        wrapper, root_dir, freeze=False, dump_samples=False, submission=False
    )
