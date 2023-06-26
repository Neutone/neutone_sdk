import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor, nn

import torchaudio
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from neutone_sdk.filters import FIRFilter, IIRFilter, FilterType

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class FilteredRAVEv1ModelWrapper(WaveformToWaveformBase):
    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        super().__init__(model, use_debug_mode)
        self.pre_filter = FIRFilter(
            FilterType.BANDPASS, cutoffs=[80.0, 4000.0], filt_size=257
        )

    def get_model_name(self) -> str:
        return "RAVE.MultiVox"

    def get_model_authors(self) -> List[str]:
        return ["Nao Tokui"]

    def get_model_short_description(self) -> str:
        return "AI model that converts any incoming sounds into voice sounds."

    def get_model_long_description(self) -> str:
        return (  # <-EDIT THIS
            "This is a stereo timbre transfer model trained on the LibriSpeech dataset, which contains the voices of various English speakers. You can adjust the formant of the output voice sound with the first knob to make it sound more like a male or female voice."
        )

    def get_technical_description(self) -> str:
        return "RAVE model proposed by Caillon, Antoine et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2111.05011",
            "Code": "https://github.com/acids-ircam/RAVE",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "RAVE", "voice"]

    def get_model_version(self) -> str:
        return "0.5.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(
                name="Randomness",
                description="How random/chaotic/unpredictable the output will be (Magnitude of latent noise)",
                default_value=0.0,
            ),
            NeutoneParameter(
                name="Formant",
                description="You can change the formant of the output voice.",
                default_value=0.5,
            ),            
            NeutoneParameter(
                name="N/A",
                description="",
                default_value=0.0,
            ),
            NeutoneParameter(
                name="N/A ", # added a space to make the key unique
                description="",
                default_value=0.0,
            ),
        ]

    def is_input_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def is_output_mono(self) -> bool:
        return True  # <-Set to False for stereo (each channel processed separately)

    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # <-Set to model sr during training

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def calc_min_delay_samples(self) -> int:
        # model latency should also be added if non-causal
        return self.pre_filter.delay

    def set_model_sample_rate_and_buffer_size(
        self, sample_rate: int, n_samples: int
    ) -> bool:
        # Set prefilter samplerate to current sample rate
        self.pre_filter.set_parameters(sample_rate=sample_rate)
        return True

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""

    @torch.no_grad()
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # Apply pre-filter
        x = self.pre_filter(x)
        ## parameters edit the latent variable
        z_mean, z_std = self.model.encode_amortized(x.unsqueeze(1))
        noise_amp = z_std * (params["Randomness"] + 0.20) * 4 # add 0.20 to avoid static noise #TODO find the reason
        batch, latent_dim, time = z_std.shape
        z = (
            torch.randn(1, latent_dim, 1, device=z_std.device).expand(batch, -1, time)
            * noise_amp
            + z_mean
        )

        # Formant knob
        female = torch.tensor([0.12933093, -0.18491942, -0.10729238, 0.11984034, 0.02838171, 0.00563526, -0.11589753, -0.00643772])
        female = female.unsqueeze(0).unsqueeze(-1)
        female *= (params["Formant"] - 0.5) * 8.0
        z += female

        out = self.model.decode(z)
        out = out.squeeze(1)
        return out


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="./models/rave/rave_cached.ts",
        help="exported RAVE torchscript file",
    )
    parser.add_argument("-o", "--output", default="ravemodel", help="model output name")
    parser.add_argument("-f", "--folder", default="./exports", help="output folder")
    parser.add_argument(
        "-s",
        "--sounds",
        nargs="*",
        type=str,
        default=None,
        help="directory of sounds to use as example input.",
    )
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    # wrap it
    model = torch.jit.load(args.input)
    wrapper = FilteredRAVEv1ModelWrapper(model)

    soundpairs = None
    if args.sounds is not None:
        soundpairs = []
        for sound in args.sounds:
            wave, sr = torchaudio.load(sound)
            input_sample = AudioSample(wave, sr)
            rendered_sample = render_audio_sample(wrapper, input_sample)
            soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    save_neutone_model(
        wrapper,
        root_dir,
        freeze=False,
        dump_samples=True,
        submission=True,
        audio_sample_pairs=soundpairs,
    )
