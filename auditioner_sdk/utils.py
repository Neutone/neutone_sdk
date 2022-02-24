from pathlib import Path
import random
from typing import Tuple
from auditioner_sdk import AuditionerModel

import torch
import json

def save_model(model: torch.jit.ScriptModule, metadata: dict, root_dir: Path):
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

  # save model and metadata!
  torch.jit.save(model, root_dir / 'model.pt')

  with open(root_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f)

def get_example_inputs(multichannel: bool = False):
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
  return [
    torch.rand((c, s)) for c, s in  zip(channels, sizes)
  ]

# import matplotlib.pyplot as plt

def test_run(model: AuditionerModel, multichannel: bool = False):
  """ 
  Performs a couple of test forward passes with audio tensors of different sizes.
  Possible inputs are audio tensors with shape (n_channels, n_samples). 
  If the model fails to meet the input/output requirements of either WaveformToWaveformBase or WaveformToLabelsBase, 
    an assertion will be triggered by the respective class. 

    Args:
      model (AuditionerModel): Your model, wrapped in either WaveformToWaveformBase or WaveformToLabelsBase
      multichannel (bool): if False, the number of input audio channels will always equal to 1. Otherwise, 
                           some stereo test input arrays will be generated.  
  Returns:
    
  """
  for x in get_example_inputs(multichannel):
    y = model(x)
    # plt.plot(y.cpu().numpy()[0])
    # plt.show()

def load_schema():
    """loads the audacity deep learning json schema for metadata"""
    from urllib.request import urlopen

    url = 'https://raw.githubusercontent.com/hugofloresgarcia/audacity/deeplearning/deeplearning-models/modelcard-schema.json'

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
  import jsonschema
  from jsonschema import validate
  schema = load_schema()

  try:
    validate(instance=metadata, schema=schema)
  except jsonschema.exceptions.ValidationError as err:
    print(err)
    return False, err

  message = "success! :)"
  return True, message