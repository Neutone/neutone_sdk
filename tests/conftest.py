import pytest
import auditioner_sdk


@pytest.fixture
def broken_metadata():
    return {"no cool": "information here"}


@pytest.fixture
def metadata():
    return {
        "sample_rate": 48000,
        "domain_tags": ["music", "speech", "environmental"],
        "short_description": "Use me to boost volume by 3dB :).",
        "long_description": "This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.",
        "tags": ["volume boost"],
        "labels": ["boosted"],
        "effect_type": "waveform-to-waveform",
        "multichannel": False,
    }


@pytest.fixture
def wav2wavmodel():
    from auditioner_sdk import WaveformToWaveformBase
    import torch
    import torch.nn as nn

    class MyVolumeModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # do the neural net magic!
            x = x * 2

            return x

    class MyVolumeModelWrapper(WaveformToWaveformBase):
        def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:

            # do any preprocessing here!
            # expect x to be a waveform tensor with shape (n_channels, n_samples)

            output = self.model(x)

            # do any postprocessing here!
            # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)

            return output

    model = MyVolumeModel()
    return MyVolumeModelWrapper(model)


@pytest.fixture
def wav2labelmodel():
    from auditioner_sdk import WaveformToLabelsBase
    import torch
    import torch.nn as nn

    class EmptyModel(nn.Module):
        def __init__(self):
            super().__init__()

    class HardCodedLabeler(WaveformToLabelsBase):
        def do_forward_pass(self, _input):
            timestamps = torch.tensor(
                [
                    [0, 0.5, 0.5, 2, 5, 2.8, 3.5, 2.5, 5.5],
                    [1, 1.5, 1.25, 3, 7, 3.5, 4, 4, 6.5],
                ]
            )
            preds = torch.tensor([0, 0, 1, 2, 0, 1, 2, 3, 0])
            return (preds, timestamps.T)

    model = EmptyModel()
    return HardCodedLabeler(model)
