import logging
import os
from typing import Dict, List

import torch as tr
from torch import Tensor

from neutone_sdk import NeutoneParameter, TextNeutoneParameter, \
    CategoricalNeutoneParameter
from neutone_sdk.non_realtime_wrapper import NonRealtimeBase

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class MusicGenModelWrapper(NonRealtimeBase):
    def get_model_name(self) -> str:
        return "MusicGen.example"

    def get_model_authors(self) -> List[str]:
        return ["Naotake Masuda"]

    def get_model_short_description(self) -> str:
        return "MusicGen model."

    def get_model_long_description(self) -> str:
        return "MusicGen model."

    def get_technical_description(self) -> str:
        return "MusicGen model."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2306.05284",
            "Code": "https://github.com/facebookresearch/audiocraft/"
        }

    def get_tags(self) -> List[str]:
        return ["musicgen"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            TextNeutoneParameter(name="prompt",
                                 description="text prompt for generation",
                                 max_n_chars=256,
                                 default_value="techno kick drum"),
            CategoricalNeutoneParameter(name="duration",
                                        description="how many seconds to generate",
                                        n_values=8,
                                        default_value=0,
                                        labels=[str(idx) for idx in range(1, 9)]),
        ]

    @tr.jit.export
    def get_audio_in_channels(self) -> List[int]:
        return []  # Does not take audio input

    @tr.jit.export
    def get_audio_out_channels(self) -> List[int]:
        return [1]  # Mono output

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [32000]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # One-shot model so buffer size does not matter

    @tr.jit.export
    def is_one_shot_model(self) -> bool:
        return True

    def do_forward_pass(self,
                        curr_block_idx: int,
                        audio_in: List[Tensor],
                        knob_params: Dict[str, Tensor],
                        text_params: List[str]) -> List[Tensor]:
        # The extra cast to int is needed for TorchScript
        n_seconds = int(knob_params["duration"].item()) + 1
        # Convert duration to number of tokens
        n_tokens = (n_seconds * 50) + 4
        if self.use_debug_mode:
            assert len(text_params) == 1
            # TorchScript does not support logging statements
            print("Preprocessing...")
        # Preprocess
        input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask = (
            self.model.preprocess(text_params, n_tokens)
        )
        # Generate
        for idx in range(n_tokens - 1):
            if self.should_cancel_forward_pass():
                return []
            input_ids = self.model.sample_step(input_ids,
                                               encoder_outputs,
                                               delay_pattern_mask,
                                               encoder_attention_mask)
            percentage_progress = int((idx + 1) / n_tokens * 100)
            self.set_progress_percentage(percentage_progress)
            if self.use_debug_mode:
                # TorchScript does not support logging statements
                print(f"Generating token {idx + 1}/{n_tokens}...")
                print(f"Progress: {self.get_progress_percentage()}%")
        if self.use_debug_mode:
            # TorchScript does not support logging statements
            print("Postprocessing...")
        # Postprocess
        audio_out = self.model.postprocess(input_ids, delay_pattern_mask, text_params)
        # Remove batch dimension
        audio_out = audio_out.squeeze(0)
        return [audio_out]


if __name__ == "__main__":
    import torchtext  # This is needed for loading the TorchScript model
    # model_path = "../../out/musicgen.ts"
    model_path = "/Users/puntland/local_christhetree/qosmo/neutone_sdk/out/musicgen.ts"
    model = tr.jit.load(model_path)
    wrapper = MusicGenModelWrapper(model)

    # TODO(cm): write export method for nonrealtime models
    # wrapper.prepare_for_inference()
    ts = tr.jit.script(wrapper)

    audio_out = wrapper.forward(curr_block_idx=0,
                                audio_in=[],
                                numerical_params=tr.tensor([0.0]).unsqueeze(1),
                                text_params=["testing"])
    log.info(audio_out[0].shape)
    audio_out = ts.forward(curr_block_idx=0,
                           audio_in=[],
                           numerical_params=tr.tensor([0.0]).unsqueeze(1),
                           text_params=["testing"])
    log.info(audio_out[0].shape)
