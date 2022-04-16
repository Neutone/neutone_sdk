import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import torch as tr
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, Parameter
from neutone_sdk.utils import model_to_torchscript, test_run, save_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# TODO(christhetree): add documentation and model validation


class ClipperModel(nn.Module):
    def forward(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        if params is None:
            min_val = -1.0
            max_val = 1.0
            gain = 1.0
            x = tr.clip(x, min=min_val*gain, max=max_val*gain)
        else:
            min_val = -params["min"].item()
            max_val = params["max"].item()
            gain = params["gain"].item()
            x = tr.clip(x, min=min_val*gain, max=max_val*gain)

        return x


class ClipperModelWrapper(WaveformToWaveformBase):
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

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_version(self) -> Union[str, int]:
        return 1

    def get_parameters(self) -> List[Parameter]:
        return [Parameter("min", "min clip threshold"),
         Parameter("max", "max clip threshold"),
         Parameter("gain", "scale clip threshold")]

    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates

    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
        x = self.model.forward(x, params)
        return x


if __name__ == "__main__":
    model = ClipperModel()
    wrapper = ClipperModelWrapper(model)
    metadata = wrapper.to_metadata_dict()
    script = model_to_torchscript(
        wrapper, freeze=True, preserved_attrs=wrapper.get_preserved_attributes()
    )

    root_dir = Path(f"exports/clipper")
    root_dir.mkdir(exist_ok=True, parents=True)
    test_run(script, multichannel=True)
    save_model(script, metadata, root_dir)

    # Check model was converted correctly
    script = tr.jit.load(root_dir / "model.pt")
    log.info(script.calc_min_delay_samples())
    log.info(script.flush())
    log.info(script.reset())
    log.info(script.set_buffer_size(512))
    log.info(json.dumps(wrapper.to_metadata_dict(), indent=4))
