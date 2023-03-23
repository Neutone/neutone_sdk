import logging
import os
from typing import Dict, List

import torch as tr
import torch.nn as nn
from torch import Tensor
from torch._C._autograd import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile
from tqdm import tqdm

from neutone_sdk import (
    WaveformToWaveformBase,
    NeutoneParameter,
    SampleQueueWrapper,
    constants,
)
from neutone_sdk.utils import model_to_torchscript

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ProfilingModel(nn.Module):
    def forward(
        self, x: Tensor, min_val: Tensor, max_val: Tensor, gain: Tensor
    ) -> Tensor:
        # tr.neg(min_val, out=min_val)
        # tr.mul(gain, min_val, out=min_val)
        # tr.mul(gain, max_val, out=max_val)
        # tr.clip(x, min=min_val, max=max_val, out=x)
        # tr.clip(x, min=gain * -min_val, max=gain * max_val, out=x)
        return x


class ProfilingModelWrapper(WaveformToWaveformBase):
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

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py"
        }

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("min", "min clip threshold", default_value=0.15),
            NeutoneParameter("max", "max clip threshold", default_value=0.15),
            NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return False

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return False

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [512]

    def get_look_behind_samples(self) -> int:
        return 0

    # def aggregate_params(self, param: Tensor) -> Tensor:
    #     return param  # We want sample-level control, so no aggregation

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        x = self.model.forward(x, min_val, max_val, gain)
        x = x[:, self.get_look_behind_samples() :]
        return x


def profile_sqw(
    sqw: SampleQueueWrapper,
    daw_sr: int = 48000,
    daw_bs: int = 512,
    daw_is_mono: bool = False,
    use_params: bool = True,
    convert_to_torchscript: bool = False,
    n_iters: int = 100,
) -> None:
    daw_n_ch = 1 if daw_is_mono else 2
    audio_buffers = [tr.rand((daw_n_ch, daw_bs)) for _ in range(n_iters)]
    if use_params:
        param_buffers = [
            tr.rand((constants.MAX_N_PARAMS, daw_bs)) for _ in range(n_iters)
        ]
    else:
        param_buffers = [None for _ in range(n_iters)]

    sqw.set_daw_sample_rate_and_buffer_size(daw_sr, daw_bs)
    sqw.prepare_for_inference()
    if convert_to_torchscript:
        log.info("Converting to TorchScript")
        with tr.no_grad():
            sqw = model_to_torchscript(sqw, freeze=False, optimize=False)

    with profile(
        activities=[ProfilerActivity.CPU],
        with_stack=True,
        profile_memory=True,
        record_shapes=False,
    ) as prof:
        with record_function("forward"):
            for audio_buff, param_buff in tqdm(zip(audio_buffers, param_buffers)):
                out_buff = sqw.forward(audio_buff, param_buff)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cpu_memory_usage", row_limit=5
        )
    )


if __name__ == "__main__":
    model = ProfilingModel()
    wrapper = ProfilingModelWrapper(model)
    sqw = SampleQueueWrapper(wrapper)
    profile_sqw(sqw)
