import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.tcn_1d import TCN1DBlock
from neutone_sdk.utils import save_neutone_model, load_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class OverdriveModel(nn.Module):
    def __init__(
            self,
            ninputs=1,
            noutputs=1,
            nblocks=4,
            channel_growth=0,
            channel_width=32,
            kernel_size=13,
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
            dilation = dilation_growth ** n
            self.blocks.append(TCN1DBlock(in_ch, out_ch, kernel_size, dilation, cond_dim=32))
            nn.init.normal_(self.blocks[-1].conv.weight)  # random initialization

        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)
        nn.init.normal_(self.output.weight)  # random initialization

    @torch.no_grad()
    def forward(self, x, c):
        p = self.condition(c)  # conditioning

        for _, block in enumerate(self.blocks):
            x = block(x, p)
        y = torch.tanh(self.output(x))  # clipping

        return y


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
        return "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. You'll get different types of distortion by re-initializing the weight or changing the activation function. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2010.04237",
            "Code": "https://github.com/csteinmetz1/ronn"
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2020). Randomized overdrive neural networks. arXiv preprint arXiv:2010.04237."

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [NeutoneParameter("P1", "Distortion 1", 0.5),
                NeutoneParameter("P2", "Distortion 2", 0.5)]

    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        p1 = params["P1"]
        p2 = params["P2"]
        condition = torch.hstack([p1, p2]).reshape((1, -1))

        # main process
        for ch in range(x.shape[0]):  # process channel by channel
            x_ = x[ch].reshape(1, 1, -1)
            x_ = self.model(x_, condition)
            x[ch] = x_.squeeze()
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
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )

    # Check model was converted correctly
    script, _ = load_neutone_model(root_dir / "model.nm")
    log.info(script.set_daw_sample_rate_and_buffer_size(48000, 2048))
    log.info(script.reset())
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))
