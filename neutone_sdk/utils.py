import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List
from neutone_sdk.audio import (
    AudioSamplePair,
    audio_sample_to_mp3_bytes,
    get_default_audio_sample,
    mp3_b64_to_audio_sample,
    render_audio_sample,
)
from neutone_sdk.core import NeutoneModel
from neutone_sdk.metadata import validate_metadata

import torch as tr
from torch import Tensor
from torch.jit import ScriptModule

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# TODO(christhetree): clean up and improve metadata validation


def model_to_torchscript(
    model: "NeutoneModel",
    freeze: bool = False,
    optimize: bool = False,
) -> ScriptModule:
    model.eval()
    script = tr.jit.script(model)
    if freeze:
        script = tr.jit.freeze(script, preserved_attrs=model.get_preserved_attributes())
    if optimize:
        log.warning(f"Optimizing may break the model.")
        script = tr.jit.optimize_for_inference(script)
    return script


def save_neutone_model(
    model: "WaveformToWaveformBase",
    root_dir: Path,
    freeze: bool = True,
    optimize: bool = False,
    dump_samples: bool = False,
    submission: bool = False,
    audio_sample_pairs: List[AudioSamplePair] = None,
) -> None:
    """
    Save a Neutone model to disk as a Torchscript file. Additionally include metadata file and samples as needed.

    Args:
        model: Your Neutone model. Should derive from neutone_sdk.WaveformToWaveformBase.
        root_dir: Directory to dump models
        dump_samples: If true, will additionally dump audio samples from the model for listening.

    Returns:
      Will create the following files:
      ```
        root_dir/
        root_dir/model.pt
        root_dir/metadata.json
        root_dir/samples/*
      ```
    """
    root_dir.mkdir(exist_ok=True, parents=True)

    script = model_to_torchscript(model, freeze=freeze, optimize=optimize)
    test_run(script)

    metadata = script.to_metadata()._asdict()
    with open(root_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    if audio_sample_pairs is None:
        input_sample = get_default_audio_sample()
        audio_sample_pairs = [
            AudioSamplePair(input_sample, render_audio_sample(script, input_sample))
        ]
    metadata["sample_sound_files"] = [
        pair.to_metadata_format() for pair in audio_sample_pairs[:3]
    ]
    validate_metadata(metadata)
    extra_files = {"metadata.json": json.dumps(metadata, indent=4).encode("utf-8")}

    # Save model and metadata
    tr.jit.save(script, root_dir / "model.nm", _extra_files=extra_files)

    if dump_samples:
        os.makedirs(root_dir / "samples", exist_ok=True)
        for i, sample in enumerate(metadata["sample_sound_files"]):
            with open(root_dir / "samples" / f"sample_in_{i}.mp3", "wb") as f:
                f.write(
                    audio_sample_to_mp3_bytes(mp3_b64_to_audio_sample(sample["in"]))
                )
            with open(root_dir / "samples" / f"sample_out_{i}.mp3", "wb") as f:
                f.write(
                    audio_sample_to_mp3_bytes(mp3_b64_to_audio_sample(sample["out"]))
                )

    if submission:  # Do extra checks
        loaded_model, loaded_metadata = load_neutone_model(root_dir / "model.nm")
        assert loaded_metadata == metadata
        del loaded_metadata["sample_sound_files"]
        assert loaded_metadata == loaded_model.to_metadata()._asdict()

        input_sample = audio_sample_pairs[0].input
        assert tr.allclose(
            render_audio_sample(model, input_sample).audio,
            render_audio_sample(loaded_model, input_sample).audio,
        )


def load_neutone_model(path: str) -> Tuple[ScriptModule, Dict]:
    extra_files = {
        "metadata.json": "",
    }
    model = tr.jit.load(path, _extra_files=extra_files)
    loaded_metadata = json.loads(extra_files["metadata.json"].decode())
    assert validate_metadata(loaded_metadata)
    return model, loaded_metadata


def get_example_inputs(multichannel: bool = False) -> List[Tensor]:
    """
    returns a list of possible input tensors for an AuditionerModel.

    Possible inputs are audio tensors with shape (n_channels, n_samples).
    If multichannel == False, n_channels will always be 1.
    """
    max_channels = 2 if multichannel else 1
    num_inputs = 10
    channels = [random.randint(1, max_channels) for _ in range(num_inputs)]
    # sizes = [random.randint(2048, 396000) for _ in range(num_inputs)]
    sizes = [2048 for _ in range(num_inputs)]
    return [tr.rand((c, s)) for c, s in zip(channels, sizes)]


def test_run(model: "NeutoneModel", multichannel: bool = False) -> None:
    """
    Performs a couple of test forward passes with audio tensors of different sizes.
    Possible inputs are audio tensors with shape (n_channels, n_samples).
    If the model fails to meet the input/output requirements of either WaveformToWaveformBase or WaveformToLabelsBase,
      an assertion will be triggered by the respective class.

      Args:
        model (NeutoneModel): Your model, wrapped in either WaveformToWaveformBase or WaveformToLabelsBase
        multichannel (bool): if False, the number of input audio channels will always equal to 1. Otherwise,
                             some stereo test input arrays will be generated.
    Returns:

    """
    for x in get_example_inputs(multichannel):
        y = model(x)
        # plt.plot(y.cpu().numpy()[0])
        # plt.show()


def validate_waveform(x: Tensor) -> None:
    assert x.ndim == 2, "input must have two dimensions (channels, samples)"
    assert x.shape[-1] > x.shape[0], (
        f"The number of channels {x.shape[-2]} exceeds the number of samples "
        f"{x.shape[-1]} in your INPUT waveform. There might be something "
        f"wrong with your model. "
    )
