import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor

import torchaudio
from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    render_audio_sample,
)

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import load_neutone_model, save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class RAVEModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "RAVE.example"

    def get_model_authors(self) -> List[str]:
        return ["Author Name"]

    def get_model_short_description(self) -> str:
        return "RAVE model trained on some data"

    def get_model_long_description(self) -> str:
        return "RAVE timbre transfer model explanation. Useful for ~ sounds."

    def get_technical_description(self) -> str:
        return "RAVE model proposed by Caillon, Antoine et al."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2111.05011",
            "Code": "https://github.com/acids-ircam/RAVE",
        }

    def get_tags(self) -> List[str]:
        return ["timbre transfer", "RAVE"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        """
        set to True for models in experimental stage
        (status shown on the website)
        """
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter(name='Chaos', description='Magnitude of latent noise', default_value=0.25),
            NeutoneParameter(name='Z edit index', description='Index of latent dimension to edit', default_value=0.0),
            NeutoneParameter(name='Z scale', description='Scale of latent variable', default_value=0.5),
            NeutoneParameter(name='Z offset', description='Offset of latent variable', default_value=0.5),
            ]

    def is_input_mono(self) -> bool:
        return True

    def is_output_mono(self) -> bool:
        return True

    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def get_citation(self) -> str:
        return """Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011."""

    @torch.no_grad()
    def do_forward_pass(
        self, x: Tensor, params: Dict[str, Tensor]
    ) -> Tensor:
        # Currently VST input-output is mono, which matches RAVE.
        if x.size(0) == 2:
            x = x.mean(dim=0, keepdim=True)

        ## parameters edit the latent variable
        z_mean, z_std = self.model.encode_amortized(x.unsqueeze(1))
        noise_amp = z_std * params['Chaos'] * 4
        z = torch.randn_like(z_std) * noise_amp + z_mean
        # add offset / scale
        idx_z = int(torch.clamp(params['Z edit index'], min=0.0, max=0.99) * self.model.cropped_latent_size)
        z_scale = params['Z scale']*2 # 0~1 -> 0~2
        z_offset = params['Z offset']*2-1 # 0~1 -> -1~1
        z[:, idx_z] = z[:, idx_z] * z_scale + z_offset
        out = self.model.decode(z)
        out = out.squeeze(1)
        
        ## Use this in case the model is wrapped in an old version and
        ## doesn't have encode_amortized
        # out = self.model.decode(self.model.encode(x.unsqueeze(1))).squeeze(1)
        return out # (n_channels=1, sample_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", default="./models/rave/rave_cached.ts", help='exported RAVE torchscript file')
    parser.add_argument("-o", "--output", default="ravemodel", help='model output name')
    parser.add_argument("-f", "--folder", default="./exports", help='output folder')
    parser.add_argument("-s", "--sounds", nargs='*', type=str, default=None, help='directory of sounds to use as example input.')
    args = parser.parse_args()
    root_dir = Path(args.folder) / args.output

    # wrap it
    model = torch.jit.load(args.input)
    wrapper = RAVEModelWrapper(model)

    soundpairs = None
    if args.sounds is not None:
        soundpairs = []
        for sound in args.sounds:
            wave, sr = torchaudio.load(sound)
            input_sample = AudioSample(wave, sr)
            rendered_sample = render_audio_sample(wrapper, input_sample)
            soundpairs.append(AudioSamplePair(input_sample, rendered_sample))

    save_neutone_model(
        wrapper, root_dir, freeze=True, dump_samples=True, submission=True, audio_sample_pairs=soundpairs
    )

    # Check model was converted correctly
    script, _ = load_neutone_model(root_dir / "model.nm")
    log.info(script.calc_min_delay_samples())
    log.info(script.flush())
    log.info(script.reset())
    log.info(script.set_buffer_size(512))
    log.info(json.dumps(wrapper.to_metadata()._asdict(), indent=4))