import io
import json
import logging
import os
import random
from pathlib import Path
import time
from typing import Tuple, Dict, List, Union

import torch as tr
from torch import Tensor
from torch.jit import ScriptModule

from neutone_sdk.audio import (
    AudioSample,
    AudioSamplePair,
    get_default_audio_samples,
    render_audio_sample,
)
from neutone_sdk.constants import MAX_N_AUDIO_SAMPLES, MAX_N_PARAMS
from neutone_sdk.core import NeutoneModel
from neutone_sdk.metadata import validate_metadata

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def dump_samples_from_metadata(metadata: Dict, root_dir: Path) -> None:
    log.info(f"Dumping samples to {root_dir/'samples'}...")
    os.makedirs(root_dir / "samples", exist_ok=True)
    for i, sample in enumerate(metadata["sample_sound_files"]):
        with open(root_dir / "samples" / f"sample_in_{i}.mp3", "wb") as f:
            f.write(AudioSample.from_b64(sample["in"]).to_mp3_bytes())
        with open(root_dir / "samples" / f"sample_out_{i}.mp3", "wb") as f:
            f.write(AudioSample.from_b64(sample["out"]).to_mp3_bytes())


def model_to_torchscript(
    model: "WaveformToWaveformBase",
    freeze: bool = False,
    optimize: bool = False,
) -> ScriptModule:
    model.prepare_for_inference()
    script = tr.jit.script(model)
    check_for_preserved_attributes(script, script.get_preserved_attributes())
    if freeze:
        log.warning(
            f"Freezing is not recommended as it may break the model or reduce TorchScript inference speed."
        )
        script = tr.jit.freeze(
            script, preserved_attrs=script.get_preserved_attributes()
        )
    if optimize:
        log.warning(f"Optimizing may break the model.")
        script = tr.jit.optimize_for_inference(script)
    check_for_preserved_attributes(script, script.get_preserved_attributes())
    return script


def save_neutone_model(
    model: ["WaveformToWaveformBase", "NonRealtimeBase"],
    root_dir: Path,
    dump_samples: bool = False,
    submission: bool = False,
    audio_sample_pairs: List[AudioSamplePair] = None,
    max_n_samples: int = MAX_N_AUDIO_SAMPLES,
    freeze: bool = False,
    optimize: bool = False,
    speed_benchmark: bool = True,
    test_offline_mode: bool = True,
) -> None:
    """
    Save a Neutone model to disk as a Torchscript file. Additionally include metadata file and samples as needed.

    Args:
        model: Your Neutone model. Should derive from neutone_sdk.WaveformToWaveformBase.
        root_dir: Directory to dump the model and auxiliary files.
        dump_samples: If true, will additionally dump audio samples from the .nm file for listening.
        submission: If true, will run additional checks to ensure the model
                saved on disk behaves identically to the one loaded in memory.
        audio_sample_pairs: Can be used to override the default samples bundled in the
                SDK to be added to the model. Will affect what is displayed on the
                website once the model is submitted.
        max_n_samples: Can be used to override the maximum number of samples
                used (default of 3) for exceptional cases.
        freeze: If true, jit.freeze will be applied to the model.
        optimize: If true, jit.optimize_for_inference will be applied to the model.
        speed_benchmark: If true, will run a speed benchmark when submission is also true.
                Consider disabling for non-realtime models as it might take too long.
        test_offline_mode: If true, will run an offline mode test. Must be true if submission is true.

    Returns:
      Will create the following files:
      ```
        root_dir/
        root_dir/model.nm
        root_dir/metadata.json
        root_dir/samples/*
      ```
    """
    if submission:
        assert dump_samples, "If submission is True then dump_samples must also be True"
        assert (
            test_offline_mode
        ), "If submission is True then test_offline_mode must also be True"

    random.seed(0)
    tr.manual_seed(0)
    if not model.use_debug_mode:
        log.warning(
            f"Debug mode has already been disabled for the model, please always test your model with debug "
            f"mode enabled."
        )

    root_dir.mkdir(exist_ok=True, parents=True)

    if model.realtime:
        from neutone_sdk.sqw import SampleQueueWrapper as Wrapper
    else:
        from neutone_sdk.non_realtime_sqw import NonRealtimeSampleQueueWrapper as Wrapper
    from neutone_sdk.benchmark import benchmark_latency_, benchmark_speed_

    wrapped = Wrapper(model)

    with tr.no_grad():
        log.info("Converting model to torchscript...")
        script = model_to_torchscript(wrapped, freeze=freeze, optimize=optimize)

        # We need to keep a copy because some models still don't implement reset
        # properly and when rendering the samples we might create unwanted state.
        #
        # We used to deepcopy but we found it breaks some models
        # script_copy = copy.deepcopy(script)
        buf = io.BytesIO()
        tr.jit.save(script, buf)
        buf.seek(0)
        script_copy = tr.jit.load(buf)

        log.info("Extracting metadata...")
        metadata = script.to_metadata()
        with open(root_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if dump_samples and not model.realtime:
            log.warning("dump_samples was true but we cannot automatically render samples for non-realtime models. Please use the Neutone Gen plugin.")
        if dump_samples and model.realtime:
            log.info("Running model on audio samples...")
            if audio_sample_pairs is None:
                input_samples = get_default_audio_samples()
                audio_sample_pairs = []
                for input_sample in input_samples:
                    # Realtime we always return only one sample
                    rendered_sample = render_audio_sample(wrapped, input_sample)[0]
                    audio_sample_pairs.append(
                        AudioSamplePair(input_sample, rendered_sample)
                    )

            metadata["sample_sound_files"] = [
                pair.to_metadata_format() for pair in audio_sample_pairs[:max_n_samples]
            ]
            log.info("Finished running model on audio samples")
        else:
            metadata["sample_sound_files"] = []

        log.info("Validating metadata...")
        validate_metadata(metadata, model.realtime)
        extra_files = {"metadata.json": json.dumps(metadata, indent=4).encode("utf-8")}

        # Save the copied model with the extra files
        log.info(f"Saving model to {root_dir/'model.nm'}...")
        tr.jit.save(script_copy, root_dir / "model.nm", _extra_files=extra_files)

        if dump_samples:
            dump_samples_from_metadata(metadata, root_dir)

        log.info("Loading saved model and metadata...")
        loaded_model, loaded_metadata = load_neutone_model(str(root_dir / "model.nm"))
        check_for_preserved_attributes(
            loaded_model, loaded_model.get_preserved_attributes()
        )
        log.info("Testing methods used by the VST...")
        loaded_model.set_daw_sample_rate_and_buffer_size(48000, 512)
        loaded_model.reset()
        loaded_model.is_resampling()

        if speed_benchmark and model.realtime:
            log.info(
                "Running speed benchmark... If this is taking too long consider "
                "disabling the speed_benchmark parameter."
            )
            benchmark_speed_(str(root_dir / "model.nm"))
            log.info("Finished speed benchmark")
        elif speed_benchmark and not model.realtime:
            log.warning("Skipping speed benchmark for non-realtime models even if speed_benchmark is True")
        else:
            log.info(
                "Skipping speed_benchmark because the speed_benchmark parameter is set "
                "to False"
            )

        if model.realtime and test_offline_mode:
            log.info("Testing offline mode...")
            for input_sample in get_default_audio_samples():
                offline_sr = input_sample.sr
                offline_bs = 4096
                loaded_model.set_daw_sample_rate_and_buffer_size(offline_sr, offline_bs)
                offline_params = tr.rand((MAX_N_PARAMS, input_sample.audio.size(1)))
                offline_rendered_sample = loaded_model.forward_offline(
                    input_sample.audio, offline_params
                )
                # TODO(cm): add comparison between online and offline rendered samples
                break  # Only test one sample to reduce export time
            log.info("Finished testing offline mode")

        if submission:  # Do extra checks
            log.info("Running submission checks...")
            log.info("Reloading model...")
            loaded_model, loaded_metadata = load_neutone_model(str(root_dir / "model.nm"))
            log.info("Assert metadata was saved correctly...")
            assert loaded_metadata == metadata
            del loaded_metadata["sample_sound_files"]
            model_metadata = loaded_model.to_metadata()
            assert loaded_metadata == model_metadata

            log.info(
                "Assert loaded model output matches output of model before saving..."
            )
            tr.manual_seed(42)
            input_sample = get_default_audio_samples()[0]
            input_sample.audio = input_sample.audio[:, :input_sample.sr * 3]
            t = time.time()
            loaded_model_render = render_audio_sample(loaded_model, input_sample)
            log.info(f"Rendering with the loaded model took {time.time()-t:.2f}s for a 3s sample and the default parameters.")
            tr.manual_seed(42)
            script_model_render = render_audio_sample(script_copy, input_sample)
            for lmr, smr in zip(loaded_model_render, script_model_render):
                assert tr.allclose(lmr.audio, smr.audio, atol=1e-6)

            log.info("Running benchmarks...")
            log.info(
                "Check out the README for additional information on how to run "
                "benchmarks with different parameters and (sample_rate, buffer_size) "
                "combinations."
            )
            log.info("Running default latency benchmark...")
            if model.realtime:
                benchmark_latency_(
                    str(root_dir / "model.nm"),
                )

            log.info("Your model has been exported successfully!")
            if model.realtime:
                log.info(
                    f"You can now test it using the FX plugin available at https://neutone.ai/fx"
                )
            else:
                log.info(
                    f"You can now test it using the Gen plugin available at https://github.com/Neutone/neutone_sdk"
                )
            log.info(
                """Additionally, the parameter helper text is not displayed
                    correctly when using the local load functionality"""
            )
            log.info(
                """If you are happy with how your model sounds and would
            like to contribute it to the default list of models, please
            consider submitting it to our GitHub. Upload only the resulting model.nm
            somewhere and open an issue on GitHub using the Request add model
            template available at the following link:"""
            )
            log.info(
                "https://github.com/QosmoInc/neutone_sdk/issues/new?assignees=bogdanteleaga%2C+christhetree&labels=enhancement&template=request-add-model.md&title=%5BMODEL%5D+%3CNAME%3E"
            )


def load_neutone_model(path: str) -> Tuple[ScriptModule, Dict]:
    extra_files = {
        "metadata.json": "",
    }
    model = tr.jit.load(path, _extra_files=extra_files)
    loaded_metadata = json.loads(extra_files["metadata.json"].decode())
    assert validate_metadata(loaded_metadata, model.realtime)
    return model, loaded_metadata


def validate_waveform(x: Tensor, is_mono: bool) -> None:
    assert x.ndim == 2, "Audio tensor must have two dimensions: (channels, samples)"
    if is_mono:
        assert (
            x.shape[0] == 1
        ), "Audio tensor should be mono and have exactly one channel"
    else:
        assert (
            x.shape[0] == 2
        ), "Audio tensor should be stereo and have exactly two channels"
    assert x.shape[-1] > x.shape[0], (
        f"The number of channels {x.shape[-2]} exceeds the number of samples "
        f"{x.shape[-1]} in the audio tensor. There might be something "
        f"wrong with the model."
    )


def check_for_preserved_attributes(
    script: ScriptModule, preserved_attrs: List[str]
) -> None:
    for attr in preserved_attrs:
        assert hasattr(script, attr), (
            f"{attr}() method is missing from the TorchScript model. Did you overwrite "
            f"it and forget to add the @torch.jit.export decorator?"
        )


def select_best_model_sr(daw_sr: int, native_sample_rates: List[int]) -> int:
    """
    Given a DAW sampling rate and a list of all the sampling rates a Neutone model supports (usually only one, or
    an empty list indicates all sampling rates are supported), determine the optimal sampling rate to use.
    """
    # Avoid resampling whenever possible
    if not native_sample_rates:
        return daw_sr
    if daw_sr in native_sample_rates:
        return daw_sr
    # Resampling is unavoidable
    if len(native_sample_rates) == 1:
        return native_sample_rates[0]
    # TODO(cm): combine this with selecting the buffer size to be smarter
    # TODO(cm): prefer upsampling if the buffer sizes allow it
    # This is a workaround for torchscript not supporting lambda functions
    diffs = [abs(sr - daw_sr) for sr in native_sample_rates]
    min_idx = diffs.index(min(diffs))
    return native_sample_rates[min_idx]


def select_best_model_buffer_size(
    io_bs: int, native_buffer_sizes: List[int]
) -> int:
    """
    Given a DAW buffer size and a list of all the buffer sizes a Neutone model supports (usually only one, or
    an empty list indicates all buffer sizes are supported), determine the optimal buffer size to use.
    """
    if not native_buffer_sizes:
        return io_bs
    if len(native_buffer_sizes) == 1:
        return native_buffer_sizes[0]
    native_buffer_sizes = sorted(native_buffer_sizes)
    for bs in native_buffer_sizes:
        if bs % io_bs == 0:
            return bs
    for bs in native_buffer_sizes:
        if bs > io_bs:
            return bs
    # TODO(cm): prefer near bs // 2 if 0 padded forward passes are enabled
    # This is a workaround for torchscript not supporting lambda functions
    diffs = [abs(bs - io_bs) for bs in native_buffer_sizes]
    min_idx = diffs.index(min(diffs))
    return native_buffer_sizes[min_idx]

