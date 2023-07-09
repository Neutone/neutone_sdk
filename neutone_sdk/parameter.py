import logging
import os
from enum import Enum
from typing import Dict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneParameterType(Enum):
    KNOB = "knob"
    BLOCK_RATE_TENSOR = "block_rate_tensor"
    AUDIO_RATE_TENSOR = "audio_rate_tensor"


class NeutoneParameter:
    """
    Define a Neutone Parameter that the user can use to control the model.

    Currently only knob type parameters are supported.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. `default_value` must be between 0 and 1 and will be used
    as a default in the plugin when no presets are available.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 dim: int = 1,
                 default_value: float = 0.0,
                 is_at_audio_rate: bool = True,
                 used: bool = True):
        self.name = name
        self.description = description
        self.dim = dim
        self.default_value = default_value
        self.is_at_audio_rate = is_at_audio_rate
        self.used = used

        # Initialize first for torchscript
        self.type = NeutoneParameterType.BLOCK_RATE_TENSOR
        if is_at_audio_rate:
            if dim == 1:
                self.type = NeutoneParameterType.KNOB
            else:
                self.type = NeutoneParameterType.AUDIO_RATE_TENSOR

    def to_metadata_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "dim": str(self.dim),
            "default_value": str(self.default_value),
            "is_at_audio_rate": str(self.is_at_audio_rate),
            "used": str(self.used),
            "type": self.type.value,
        }
