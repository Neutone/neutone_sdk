import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import torch as tr
from torch import Tensor, nn

from neutone_sdk import WaveformToWaveformBase, Parameter
from neutone_sdk.realtime_stft import RealtimeSTFT
from neutone_sdk.utils import test_run, save_model, model_to_torchscript

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

SR = 44100
# SR = 48000


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
    def get_model_name(self) -> str:
        return 'spectral_diff.test'

    def get_model_authors(self) -> List[str]:
        return ['Christopher Mitcheltree']

    def get_model_short_description(self) -> str:
        return "Neural spectral distortion effect"

    def get_model_long_description(self) -> str:
        return "A neural autoencoder reconstructs the spectrogram of the signal which is then subtracted from the original signal's spectrogram."

    def get_technical_description(self) -> str:
        return "TBD once tests are complete."

    def get_tags(self) -> List[str]:
        return ['spectral', 'distortion', 'autoencoder', 'high-pass filter']

    def get_version(self) -> Union[str, int]:
        return 1

    def get_parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name='Spectral dry/wet',
                description="Control how much of the reconstructed spectrogram is subtracted from the original signal's spectrogram.",
            )
        ]

    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False

    def get_native_sample_rates(self) -> List[int]:
        return [SR]

    def get_native_buffer_sizes(self) -> List[int]:
        # This model has a maximum buffer size and requires buffer sizes of
        # specific multiples
        return self.model.calc_supported_buffer_sizes()

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        return self.model.calc_min_delay_samples()

    @tr.jit.export
    def set_buffer_size(self, n_samples: int) -> bool:
        self.model.set_buffer_size(n_samples)
        return True

    @tr.jit.export
    def flush(self) -> Optional[Tensor]:
        if self.model.calc_min_delay_samples() == 0:
            return None
        else:
            return self.model.flush()

    @tr.jit.export
    def reset(self) -> bool:
        self.model.reset()
        return True

    def do_forward_pass(self,
                        x: Tensor,
                        params: Optional[Dict[str, Tensor]] = None) -> Tensor:
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
    metadata = wrapper.to_metadata_dict()
    script = model_to_torchscript(
        wrapper,
        freeze=True,
        preserved_attrs=wrapper.get_preserved_attributes()
    )

    root_dir = Path(f'../exports/spectral_diff__sr_{SR}')
    root_dir.mkdir(exist_ok=True, parents=True)
    test_run(script, multichannel=True)
    save_model(script, metadata, root_dir)

    # Check model was converted correctly
    script = tr.jit.load(root_dir / 'model.pt')
    log.info(script.calc_min_delay_samples())
    log.info(script.flush())
    log.info(script.reset())
    log.info(script.set_buffer_size(512))
    log.info(json.dumps(wrapper.to_metadata_dict(), indent=4))
