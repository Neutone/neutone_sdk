import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import NeutoneParameter, ContinuousNeutoneParameter
from neutone_sdk.non_realtime_wrapper import NonRealtimeBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ClipperModel(nn.Module):
    def forward(self,
                x: Tensor,
                min_val: Tensor,
                max_val: Tensor,
                gain: Tensor) -> Tensor:
        tr.neg(min_val, out=min_val)
        tr.mul(gain, min_val, out=min_val)
        tr.mul(gain, max_val, out=max_val)
        tr.clip(x, min=min_val, max=max_val, out=x)
        return x


class NonRealtimeClipperModelWrapper(NonRealtimeBase):
    def get_model_name(self) -> str:
        return "clipper"

    def get_model_authors(self) -> List[str]:
        return ["Christopher Mitcheltree"]

    def get_model_short_description(self) -> str:
        return "Audio clipper."

    def get_model_long_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/neutone_gen/example_clipper_gen.py"
        }

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter("min", "min clip threshold", default_value=0.15),
            ContinuousNeutoneParameter("max", "max clip threshold", default_value=0.15),
            ContinuousNeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @tr.jit.export
    def get_audio_in_channels(self) -> List[int]:
        return [2]

    @tr.jit.export
    def get_audio_out_channels(self) -> List[int]:
        return [2]

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    @tr.jit.export
    def is_one_shot_model(self) -> bool:
        return False

    def aggregate_continuous_params(self, cont_params: Tensor) -> Tensor:
        return cont_params  # We want sample-level control, so no aggregation

    def do_forward_pass(self,
                        curr_block_idx: int,
                        audio_in: List[Tensor],
                        knob_params: Dict[str, Tensor],
                        text_params: List[str]) -> List[Tensor]:
        min_val, max_val, gain = (knob_params["min"],
                                  knob_params["max"],
                                  knob_params["gain"])
        audio_out = []
        for x in audio_in:
            x = self.model.forward(x, min_val, max_val, gain)
            audio_out.append(x)
        return audio_out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    model = ClipperModel()
    wrapper = NonRealtimeClipperModelWrapper(model)

    # TODO(cm): write export method for nonrealtime models
    wrapper.forward(0, [tr.rand(2, 2048)])
    ts = tr.jit.script(wrapper)
    ts.forward(0, [tr.rand(2, 2048)])
