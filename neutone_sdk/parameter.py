import logging
import os
from enum import Enum
from typing import NamedTuple, Dict, Any

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class ParameterType(Enum):
    KNOB = 'knob'


class Parameter(NamedTuple):
    name: str
    description: str
    type: ParameterType = ParameterType.KNOB
    used: bool = True

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'type': self.type.value,
            'used': self.used,
        }
