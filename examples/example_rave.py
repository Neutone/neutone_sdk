import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class RAVEModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "RAVE.jvoice"

    def get_model_authors(self) -> List[str]:
        return ["Nao Tokui"]

    def get_model_short_description(self) -> str:
        return "RAVE model trained on Japanese female voice"

    def get_model_long_description(self) -> str:
        return "RAVE timbre transfer model trained on Japanese female voice"

    def get_technical_description(self) -> str:
        return "RAVE model proposed by Caillon Antoine et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2111.05011",
            "Code": "https://github.com/acids-ircam/RAVE",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "voice"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return []

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""

    @torch.no_grad()
    def do_forward_pass(
        self, x: Tensor, params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        # Currently VST input-output is mono, which matches RAVE.
        if x.size(0) == 2:
            x = x.mean(dim=0, keepdim=True)
        x = self.model(x.unsqueeze(0)).squeeze(1)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default="./models/rave/rave_jsut_cached.ts")
    parser.add_argument("-o", "--output", default="./exports/rave_jsut.pt")
    args = parser.parse_args()
    root_dir = Path(args.output)

    # wrap it
    model = torch.jit.load(args.input)
    wrapper = RAVEModelWrapper(model)
    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )

    # Check model was converted correctly
    script, _ = load_neutone_model(root_dir / "model.nm")
    log.info(script.calc_min_delay_samples())
    log.info(script.flush())
    log.info(script.reset())
    log.info(script.set_buffer_size(512))
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))
