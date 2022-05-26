import copy
import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List
from neutone_sdk.audio import (
    AudioSamplePair,
    audio_sample_to_mp3_bytes,
    get_default_audio_samples,
    mp3_b64_to_audio_sample,
    render_audio_sample,
)
from neutone_sdk.constants import MAX_N_AUDIO_SAMPLES
from neutone_sdk.core import NeutoneModel
from neutone_sdk.metadata import validate_metadata

import torch as tr
from torch import Tensor
from torch.jit import ScriptModule

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def dump_samples_from_metadata(metadata: Dict, root_dir: Path) -> None:
    log.info(f"Dumping samples to {root_dir/'samples'}...")
    os.makedirs(root_dir / "samples", exist_ok=True)
    for i, sample in enumerate(metadata["sample_sound_files"]):
        with open(root_dir / "samples" / f"sample_in_{i}.mp3", "wb") as f:
            f.write(audio_sample_to_mp3_bytes(mp3_b64_to_audio_sample(sample["in"])))
        with open(root_dir / "samples" / f"sample_out_{i}.mp3", "wb") as f:
            f.write(audio_sample_to_mp3_bytes(mp3_b64_to_audio_sample(sample["out"])))


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
    max_n_samples: int = MAX_N_AUDIO_SAMPLES,
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

    with tr.no_grad():
        log.info("Converting model to torchscript...")
        script = model_to_torchscript(model, freeze=freeze, optimize=optimize)
        # We need to keep a copy because some models still don't implement reset
        # properly and when rendering the samples we might create unwanted state.
        script_copy = copy.deepcopy(script)

        log.info("Extracting metadata...")
        metadata = script.to_metadata()._asdict()
        with open(root_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        log.info("Running model on audio samples...")
        if audio_sample_pairs is None:
            input_samples = get_default_audio_samples()
            audio_sample_pairs = []
            for input_sample in input_samples:
                rendered_sample = render_audio_sample(model, input_sample)
                audio_sample_pairs.append(
                    AudioSamplePair(input_sample, rendered_sample)
                )

        metadata["sample_sound_files"] = [
            pair.to_metadata_format() for pair in audio_sample_pairs[:max_n_samples]
        ]
        log.info("Validating metadata...")
        validate_metadata(metadata)
        extra_files = {"metadata.json": json.dumps(metadata, indent=4).encode("utf-8")}

        # Save the copied model with the extra files
        log.info(f"Saving model to {root_dir/'model.nm'}...")
        tr.jit.save(script_copy, root_dir / "model.nm", _extra_files=extra_files)

        if dump_samples:
            dump_samples_from_metadata(metadata, root_dir)

        if submission:  # Do extra checks
            log.info("Running submission checks...")
            log.info("Loading saved model and metadata...")
            loaded_model, loaded_metadata = load_neutone_model(root_dir / "model.nm")
            log.info("Assert metadata was saved correctly...")
            assert loaded_metadata == metadata
            del loaded_metadata["sample_sound_files"]
            assert loaded_metadata == loaded_model.to_metadata()._asdict()

            log.info(
                "Assert loaded model output matches output of model before saving..."
            )
            input_samples = audio_sample_pairs[0].input
            tr.manual_seed(42)
            loaded_model_render = render_audio_sample(loaded_model, input_samples).audio
            tr.manual_seed(42)
            script_model_render = render_audio_sample(script_copy, input_samples).audio

            assert tr.allclose(script_model_render, loaded_model_render)

            log.info("Your model has been exported successfully!")
            log.info(
                "You can now test it using the plugin available at https://neutone.space"
            )
            log.info("Note that in beta we only support 48kHz SR / 2048 buffer size")
            log.info(
                """Additionally, the parameter helper text is not displayed
                    correctly when using the local load functionality"""
            )
            log.info(
                """If you are happy with how your model sounds and would
            like to contribute it to the default list of models, please
            consider submitting it to our GitHub. Upload the resulting model.nm
            somewhere and open an issue on GitHub using the Request add model
            template available at the following link:"""
            )
            log.info(
                "https://github.com/QosmoInc/neutone-sdk/issues/new?assignees=bogdanteleaga%2C+christhetree&labels=enhancement&template=request-add-model.md&title=%5BMODEL%5D+%3CNAME%3E"
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
