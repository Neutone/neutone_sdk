import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from urllib.request import urlopen

import torch as tr
from jsonschema import validate, ValidationError
from torch import Tensor, nn
from torch.jit import ScriptModule

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# TODO(christhetree): clean up and improve metadata validation


def model_to_torchscript(
    model: nn.Module,
    freeze: bool = False,
    preserved_attrs: Optional[List[str]] = None,
    optimize: bool = False,
) -> ScriptModule:
    model.eval()
    script = tr.jit.script(model)
    if freeze:
        script = tr.jit.freeze(script, preserved_attrs=preserved_attrs)
    if optimize:
        log.warning(f"Optimizing may break the model.")
        script = tr.jit.optimize_for_inference(script)
    return script


def save_model(model: ScriptModule, metadata: Dict[str, Any], root_dir: Path) -> None:
    """
    Save a compiled torch.jit.ScriptModule, along with a metadata dictionary.

    Args:
        model: your Auditioner-ready serialized model, using either torch.jit.trace or torch.jit.script.
          Should derive from auditioner_sdk.WaveformToWaveformBase or auditioner_sdk.WaveformToLabelsBase.
        metadata: a metadata dictionary. Shoule be validated using torchaudio.utils.validate_metadata()

    Returns:
      Will create the following files:
      ```
        root_dir/
        root_dir/model.pt
        root_dir/metadata.json
      ```
    """
    root_dir.mkdir(exist_ok=True, parents=True)

    # Save model and metadata
    tr.jit.save(model, root_dir / "model.pt")

    with open(root_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


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


def load_schema() -> Dict:
    """loads the audacity deep learning json schema for metadata"""
    url = "https://raw.githubusercontent.com/hugofloresgarcia/audacity/deeplearning/deeplearning-models/modelcard-schema.json"
    response = urlopen(url)
    schema = json.loads(response.read())
    return schema


def validate_metadata(metadata: dict) -> Tuple[bool, str]:
    """validate a model metadata dict using Auditioner's metadata schema

    Args:
        metadata (dict): the metadata dictionary to validate

    Returns:
        Tuple[bool, str], where the  bool indicates success, and
        the string contains an error/success message
    """
    schema = load_schema()

    try:
        validate(instance=metadata, schema=schema)
    except ValidationError as err:
        log.info(err)
        return False, str(err)

    message = "success! :)"
    return True, message


def validate_waveform(x: Tensor) -> None:
    assert x.ndim == 2, "input must have two dimensions (channels, samples)"
    assert x.shape[-1] > x.shape[0], (
        f"The number of channels {x.shape[-2]} exceeds the number of samples "
        f"{x.shape[-1]} in your INPUT waveform. There might be something "
        f"wrong with your model. "
    )
