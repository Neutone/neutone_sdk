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
    name: str
    description: str
    type: NeutoneParameterType = NeutoneParameterType.KNOB
    used: bool = True

    def to_metadata_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "used": str(self.used),
        }
