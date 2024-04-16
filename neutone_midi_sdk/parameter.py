import logging
import os
from abc import ABC
from enum import Enum
import torch
from typing import Dict, Optional, Set, List, Union

from neutone_midi_sdk import constants

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

class NeutoneParameterType(Enum):
    BASE = "base"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TENSOR = "tensor"
    
class NeutoneParameter(ABC):
    """
    Defines a Neutone Parameter abstract base class.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. This parameter has no functionality and is meant to subclassed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        default_value: Union[int, float, str], #TODO(nic): optional default_value for tensor case, or default to uniformly populating tensor with default_value 
        used: bool,
        param_type: NeutoneParameterType,
    ):
        self.name = name
        self.description = description
        self.default_value = default_value
        self.used = used
        self.type = param_type

    def to_metadata_dict(self) -> Dict[str, str]:
        """Returns a string dictionary containing the metadata of the parameter."""
        return {
            "name": self.name,
            "description": self.description,
            "default_value": str(self.default_value),
            "used": str(self.used),
            "type": self.type.value,
        }

class ContinuousNeutoneParameter(NeutoneParameter):
    """
    Defines a continuous Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `default_value` must be between 0 and 1 and will be used as a default in the plugin
    when no presets are available.
    """

    def __init__(
        self, name: str, description: str, default_value: float, used: bool = True
    ):
        super().__init__(
            name,
            description,
            default_value,
            used,
            NeutoneParameterType.CONTINUOUS,
        )
        assert (
            0.0 <= default_value <= 1.0
        ), "`default_value` for continuous params must be between 0 and 1"








    
# class TensorNeutoneParameter(NeutoneParameter):
#     """
#     Defines a tensor Neutone Parameter that the user can use to control a model.
#     """
#     def __init__(self,
#                  name: str,
#                  description: str,
#                  shape: List[int],
#                  default_tensor: Optional[torch.Tensor] = None,
#                  used: bool = True):
#         super().__init__(name, description, used)
#         self.shape = shape
#         self.default_tensor = default_tensor
#         self.type = NeutoneParameterType.TENSOR

#     def to_metadata_dict(self) -> Dict[str, str]:
#         """Returns a string dictionary containing the metadata of the parameter."""
#         data = super().to_metadata_dict()
#         data["shape"] = str(self.shape)
#         data["default_tensor"] = str(self.default_tensor.numpy())
#         return data







# from enum import Enum
# from typing import Dict


# class NeutoneParameterType(Enum):
#     KNOB = "knob"
#     AUX_INPUT_TENSOR = "aux_input_tensor"

# class NeutoneParameter:
#     """
#     Define a Neutone Parameter that the user can use to control the model.

#     The name and the description of the parameter will be shown as a tooltip
#     within the UI. `default_value` must be between 0 and 1 and will be used
#     as a default in the plugin when no presets are available.
#     """

#     def __init__(
#             self,
#             name: str,
#             description: str,
#             dim: int = 1,
#             default_value: list = [0.0], # TODO: adjust type here
#             used: bool = True
#         ):
    

#         self.name = name
#         self.description = description
#         self.dim = dim
#         self.default_value = default_value
#         self.used = used

#         if self.dim == 1:
#             self.type = NeutoneParameterType.KNOB
#         else:
#             self.type = NeutoneParameterType.AUX_INPUT_TENSOR
        
#     def to_metadata_dict(self) -> Dict[str, str]:
#         return {
#             "name": self.name,
#             "description": self.description,
#             "type": self.type.value,
#             "used": str(self.used),
#             "default_value": str(self.default_value),
#         }