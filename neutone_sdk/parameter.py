import logging
import os
from abc import ABC
from enum import Enum
from typing import Union, Optional, List, Dict

from torch import Tensor as T, nn
import torch as tr
from neutone_sdk import constants

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneParameterType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TOKENS = "tokens"


class NeutoneParameter(ABC, nn.Module):
    """
    Defines a Neutone Parameter abstract base class.

    The name and the description of the parameter will be shown as a tooltip
    within the UI. This parameter has no functionality and is meant to subclassed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        default_value: Union[int, float, str, Optional[List[int]]],
        used: bool,
        param_type: NeutoneParameterType,
    ):
        super().__init__()
        self.name = name
        self.description = description
        self.default_value = default_value
        self.used = used
        self.type = param_type

    def to_metadata(
        self,
    ) -> Dict[str, Union[int, float, str, bool, List[str], List[int]]]:
        return {
            "name": self.name,
            "description": self.description,
            "default_value": self.default_value,
            "used": self.used,
            "type": self.type.value,
        }


class ContinuousNeutoneParameter(NeutoneParameter):
    """
    Defines a continuous Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `default_value` must be between min_value and max_value and will be used as the
    default in the plugin when no presets are available.
    """

    def __init__(
        self,
        name: str,
        description: str,
        default_value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
        used: bool = True,
    ):
        super().__init__(
            name,
            description,
            default_value,
            used,
            NeutoneParameterType.CONTINUOUS,
        )
        assert (
            min_value < max_value
        ), "`min_value` must be less than `max_value` for continuous params"
        assert (
            min_value <= default_value <= max_value
        ), f"`default_value` for continuous params must be between {min_value} and {max_value}"
        self.min_value = min_value
        self.max_value = max_value
        self.range = max_value - min_value
        self.default_value_0to1 = (default_value - min_value) / self.range

    def from_0to1(self, param_val: T) -> T:
        """
        Converts a parameter value inplace from [0, 1] to [min_value, max_value].
        """
        tr.mul(param_val, self.range, out=param_val)
        tr.add(param_val, self.min_value, out=param_val)
        return param_val

    def to_metadata(self) -> Dict[str, Union[int, float, str, bool, List[str]]]:
        metadata = super().to_metadata()
        metadata["min_value"] = self.min_value
        metadata["max_value"] = self.max_value
        return metadata


class CategoricalNeutoneParameter(NeutoneParameter):
    """
    Defines a categorical Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `n_values` must be an int greater than or equal to 2 and less than or equal to
    `constants.MAX_N_CATEGORICAL_VALUES`.
    `default_value` must be in the range [0, `n_values` - 1].
    `labels` is a list of strings that will be used as the labels for the parameter.
    """

    def __init__(
        self,
        name: str,
        description: str,
        n_values: int,
        default_value: int,
        labels: Optional[List[str]] = None,
        used: bool = True,
    ):
        super().__init__(
            name, description, default_value, used, NeutoneParameterType.CATEGORICAL
        )
        assert 2 <= n_values <= constants.MAX_N_CATEGORICAL_VALUES, (
            f"`n_values` for categorical params must between 2 and "
            f"{constants.MAX_N_CATEGORICAL_VALUES}"
        )
        assert (
            0 <= default_value <= n_values - 1
        ), "`default_value` for categorical params must be between 0 and `n_values`-1"
        self.n_values = n_values
        if labels is None:
            labels = [str(idx) for idx in range(n_values)]
        else:
            assert len(labels) == self.n_values, "labels must have `n_values` elements"
        assert all(
            len(label) < constants.MAX_N_CATEGORICAL_LABEL_CHARS for label in labels
        ), (
            f"All labels must have length less than "
            f"{constants.MAX_N_CATEGORICAL_LABEL_CHARS} characters"
        )
        self.labels = labels
        self.default_value_0to1 = default_value / (n_values - 1)

    def from_0to1(self, param_val: T) -> T:
        """
        Converts a parameter value inplace from [0, 1] to [0, `n_values` - 1].
        """
        tr.mul(param_val, self.n_values - 1, out=param_val)
        tr.round(param_val, out=param_val)
        return param_val

    def to_metadata(
        self,
    ) -> Dict[str, Union[int, float, str, bool, List[str], List[int]]]:
        metadata = super().to_metadata()
        metadata["n_values"] = self.n_values
        metadata["labels"] = self.labels
        return metadata


class TextNeutoneParameter(NeutoneParameter):
    """
    Defines a text Neutone Parameter that the user can use to control a model.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    `max_n_chars` specifies the maximum number of characters that the user can input.
    If this value is set to -1, there is no limit on the number of characters.
    `default_value` is the default value to be automatically populated in the text box.
    """

    def __init__(
        self,
        name: str,
        description: str,
        max_n_chars: int = -1,
        default_value: str = "",
        used: bool = True,
    ):
        super().__init__(
            name, description, default_value, used, NeutoneParameterType.TEXT
        )
        assert max_n_chars >= -1, "`max_n_chars` must be greater than or equal to -1"
        if max_n_chars != -1:
            assert (
                len(default_value) <= max_n_chars
            ), "`default_value` must be a string of length less than `max_n_chars`"
        self.max_n_chars = max_n_chars

    def to_metadata(
        self,
    ) -> Dict[str, Union[int, float, str, bool, List[str], List[int]]]:
        metadata = super().to_metadata()
        metadata["max_n_chars"] = self.max_n_chars
        return metadata


class DiscreteTokensNeutoneParameter(NeutoneParameter):
    """
    Defines a discrete token tensor input to a Neutone model
    Should be the output of a tokenizer that processes some text input.

    The name and the description of the parameter will be shown as a tooltip
    within the UI.
    """

    def __init__(
        self,
        name: str,
        description: str,
        max_n_tokens: int = -1,
        default_value: Optional[List[int]] = None,
        used: bool = True,
    ):
        if default_value is None:
            default_value: List[int] = []
        super().__init__(
            name, description, default_value, used, NeutoneParameterType.TOKENS
        )
        assert max_n_tokens >= -1, "`max_n_tokens` must be greater than or equal to -1"
        if max_n_tokens != -1:
            assert (
                len(default_value) <= max_n_tokens
            ), "`default_value` must be a list of length less than `max_n_tokens`"
        self.max_n_tokens = max_n_tokens

    def to_metadata(
        self,
    ) -> Dict[str, Union[int, float, str, bool, List[str], List[int]]]:
        metadata = super().to_metadata()
        metadata["max_n_tokens"] = self.max_n_tokens
        return metadata
