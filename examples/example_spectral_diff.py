import logging
import os
from pathlib import Path
from typing import Tuple

import torch as tr
from auditioner_sdk.utils import test_run, validate_metadata, save_model, \
    model_to_torchscript
from torch import Tensor, nn

from auditioner_sdk import WaveformToWaveformBase
from auditioner_sdk.realtime_stft import RealtimeSTFT

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

SR = 44100

metadata = {
    "model_name": "spectral_diff.test",
    "model_authors": [
        "Christopher Mitcheltree"
    ],
    "model_short_description": "Neural spectral distortion effect",
    "model_long_description": "A neural autoencoder reconstructs the spectrogram of the signal which is then subtracted from the original signal's spectrogram.",
    "technical_description": "TBD once tests are complete.",
    "technical_links": {
        "Paper": "",
        "Code": "",
    },
    "tags": [
        "spectral",
        "distortion",
        "autoencoder",
        "high-pass filter",
    ],
    "model_type": "stereo-stereo",
    "sample_rate": SR,
    "minimum_buffer_size": 512,
    "parameters": {
        "p1": {
            "used": True,
            "name": "Spectral dry/wet",
            "description": "Control how much of the reconstructed spectrogram is subtracted from the original signal's spectrogram.",
            "type": "knob"
        },
        "p2": {
            "used": False,
            "name": "",
            "description": "",
            "type": "knob"
        },
        "p3": {
            "used": False,
            "name": "",
            "description": "",
            "type": "knob"
        },
        "p4": {
            "used": False,
            "name": "",
            "description": "",
            "type": "knob"
        }
    },
    "version": 1,

    # TODO(christhetree): remove
    "domain_tags": ["music"],
    "short_description": "",
    "long_description": "",
    "labels": ["spectral", "distortion", "autoencoder", "high-pass filter"],
    "effect_type": "waveform-to-waveform",
    "multichannel": True,
}


class SpecCNN2DSmall(nn.Module):
    def __init__(self,
                 n_filters: int = 4,
                 kernel: Tuple[int] = (5, 5),
                 pooling: Tuple[int] = (4, 2),
                 activation: nn.Module = nn.ELU()) -> None:
        super().__init__()
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.enc = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv2d(n_filters, n_filters * 4, kernel, stride=pooling, padding=padding),
            activation,
            nn.Conv2d(n_filters * 4, n_filters * 16, kernel, stride=pooling, padding=padding),
            activation,
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(n_filters * 16, n_filters * 4, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters * 4, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, n_filters, kernel, stride=pooling, padding=padding, output_padding=(0, 1)),
            activation,
            nn.ConvTranspose2d(n_filters, 1, kernel, stride=(1, 1), padding=padding),
        )

    def forward(self, spec: Tensor) -> Tensor:
        spec = spec.unsqueeze(1)
        z = self.enc(spec)
        rec = self.dec(z)
        rec = rec.squeeze(1)
        return rec


class SpectralDiffWrapper(WaveformToWaveformBase):
    def do_forward_pass(self, x: Tensor) -> Tensor:
        return self.model.forward(x)


if __name__ == '__main__':
    models_dir = '../models/'
    if SR == 44100:
        model_weights_name = 'SpecCNN2DSmall__sr_44100__n_fft_2048__center_True__n_frames_16__pos_spec_False__n_filters_4__epoch=04__val_loss=0.298.pt'
    elif SR == 48000:
        model_weights_name = 'SpecCNN2DSmall__sr_48000__n_fft_2048__center_True__n_frames_16__pos_spec_False__n_filters_4__epoch=04__val_loss=0.340.pt'
    else:
        raise ValueError
    log.info('Loading weights')
    model_path = os.path.join(models_dir, model_weights_name)

    n_filters = 4
    model = SpecCNN2DSmall(n_filters=n_filters)
    model.load_state_dict(tr.load(model_path, map_location=tr.device('cpu')))

    rts = RealtimeSTFT(
        model=model,
        batch_size=2,  # Stereo
        io_n_samples=2048,
        n_fft=2048,
        hop_len=512,
        model_io_n_frames=16,
        center=True,
        spec_diff_mode=True,
        power=1.0,
        logarithmize=True,
        ensure_pos_spec=False,
        use_phase_info=True,
        fade_n_samples=32,
    )
    wrapper = SpectralDiffWrapper(rts)
    script = model_to_torchscript(wrapper, freeze=True, optimize=True)

    root_dir = Path(f'../exports/spectral_diff__sr_{SR}')
    root_dir.mkdir(exist_ok=True, parents=True)
    test_run(script, multichannel=True)
    success, msg = validate_metadata(metadata)
    assert success
    save_model(script, metadata, root_dir)
