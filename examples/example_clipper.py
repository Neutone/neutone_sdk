import json
import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Optional, Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ClipperModel(nn.Module):
    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        if params is None:
            min_val = -1.0
            max_val = 1.0
            gain = 1.0
            x = tr.clip(x, min=min_val * gain, max=max_val * gain)
        else:
            for i in range(x.shape[-1]):
                min_val = -params["min"][i]
                max_val = params["max"][i]
                gain = params["gain"][i]
                for c in range(x.shape[0]):
                    x[c][i] = tr.min(tr.max(x[c][i], gain * min_val), gain * max_val)
        return x


class ClipperModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "clipper"

    def get_model_authors(self) -> List[str]:
        return ["Andrew Fyfe"]

    def get_model_short_description(self) -> str:
        return "Audio clipper."

    def get_model_long_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_params(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("min", "min clip threshold", default_value=1.0),
            NeutoneParameter("max", "max clip threshold", default_value=1.0),
            NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def aggregate_param(self, param: Tensor) -> Tensor:
        assert param.ndim == 1
        return param  # We want sample-level control, so no aggregation

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        x = self.model.forward(x, params)
        return x


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = ClipperModel()
    wrapper = ClipperModelWrapper(model)
    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True
    )

    script, _ = load_neutone_model(root_dir / "model.nm")

    # Check model was converted correctly
    log.info(script.calc_min_delay_samples())
    log.info(script.flush())
    log.info(script.reset())
    log.info(script.set_daw_sample_rate_and_buffer_size(44100, 512))
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))
