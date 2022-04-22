import logging
import os
from enum import Enum
from typing import NamedTuple, Dict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneParameterType(Enum):
    KNOB = "knob"


class NeutoneParameter(NamedTuple):
    """
    Define a Neutone Parameter that the user can use to control the model.

    Currently only knob type parameters are supported.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. `default_value` must be between 0 and 1 and will be used
    as a default in the plugin when no presets are available.
    """

    name: str
    description: str
    default_value: float
    type: NeutoneParameterType = NeutoneParameterType.KNOB
    used: bool = True

    def to_metadata_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "used": str(self.used),
            "default_value": str(self.default_value),
        }
