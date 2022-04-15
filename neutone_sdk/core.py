import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional, Union, Final

import torch as tr
from torch import nn, Tensor

from neutone_sdk.parameter import Parameter
from neutone_sdk.utils import validate_waveform

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneModel(ABC, nn.Module):
    MAX_N_PARAMS: Final[int] = 4

    model_name: Final[str]
    model_authors: Final[List[str]]
    model_short_description: Final[str]
    model_long_description: Final[str]
    technical_description: Final[str]
    tags: Final[List[str]]
    version: Final[Union[str, int]]
    technical_paper_link: Final[Optional[str]]
    technical_code_link: Final[Optional[str]]

    def __init__(self, model: nn.Module) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()
        assert len(self.get_parameters()) <= NeutoneModel.MAX_N_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_parameters()])) == len(
            self.get_parameters()
        )
        model.eval()
        self.model = model

        self.model_name = self.get_model_name()
        self.model_authors = self.get_model_authors()
        self.model_short_description = self.get_model_short_description()
        self.model_long_description = self.get_model_long_description()
        self.technical_description = self.get_technical_description()
        self.tags = self.get_tags()
        self.version = self.get_version()
        self.technical_paper_link = self.get_technical_paper_link()
        self.technical_code_link = self.get_technical_code_link()

        # TODO(christhetree): check all preserved_attrs have been exported

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_model_authors(self) -> List[str]:
        pass

    @abstractmethod
    def get_model_short_description(self) -> str:
        pass

    @abstractmethod
    def get_model_long_description(self) -> str:
        pass

    @abstractmethod
    def get_technical_description(self) -> str:
        pass

    @abstractmethod
    def get_tags(self) -> List[str]:
        pass

    @abstractmethod
    def get_version(self) -> Union[str, int]:
        pass

    def get_technical_paper_link(self) -> Optional[str]:
        return None

    def get_technical_code_link(self) -> Optional[str]:
        return None

    def get_parameters(self) -> List[Parameter]:
        return []

    def get_preserved_attributes(self) -> List[str]:
        return []

    def to_metadata_dict(self) -> Dict[str, Any]:
        metadata_dict = {
            "model_name": self.model_name,
            "model_authors": self.model_authors,
            "model_short_description": self.model_short_description,
            "model_long_description": self.model_long_description,
            "technical_description": self.technical_description,
            "technical_links": {
                "Paper": self.technical_paper_link or "",
                "Code": self.technical_code_link or "",
            },
            "tags": self.tags,
            "version": self.version,
        }
        parameters_dict = {}
        parameters = self.get_parameters()
        # Ensure there are always MAX_N_PARAMS in the metadata json
        for idx in range(NeutoneModel.MAX_N_PARAMS):
            k = f"p{idx + 1}"
            if idx < len(parameters):
                v = parameters[idx].to_metadata_dict()
            else:
                v = Parameter(used=False, name="", description="").to_metadata_dict()
            parameters_dict[k] = v
        metadata_dict["parameters"] = parameters_dict
        return metadata_dict


class WaveformToLabelsBase(NeutoneModel):
    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        Internal forward pass for a WaveformToLabels model.

        All this does is wrap the do_forward_pass(x) function in assertions that check
        that the correct input/output constraints are getting met. Nothing fancy.
        """
        validate_waveform(x)
        output = self.do_forward_pass(x)

        assert isinstance(output, tuple), "waveform-to-labels output must be a tuple"
        assert (
            len(output) == 2
        ), "output tuple must have two elements, e.g. tuple(labels, timestamps)"

        labels = output[0]
        timestamps = output[1]

        assert tr.all(
            timestamps >= 0
        ).item(), f"found a timestamp that is less than zero"

        for timestamp in timestamps:
            assert (
                timestamp[0] < timestamp[1]
            ), f"timestamp ends ({timestamp[1]}) before it starts ({timestamp[0]})"

        assert labels.ndim == 1, "labels tensor should be one dimensional"

        assert (
            labels.shape[0] == timestamps.shape[0]
        ), "time dimension between labels and timestamps tensors must be equal"
        assert (
            timestamps.shape[1] == 2
        ), "second dimension of the timestamps tensor must be size 2"
        return output

    def do_forward_pass(self, x: Tensor) -> (Tensor, Tensor):
        """
        Perform a forward pass on a waveform-to-labels model.

        Args:
            x : An input audio waveform tensor. If `"multichannel" == True` in the
                model's `metadata.json`, then this tensor will always be shape
                `(1, n_samples)`, as all incoming audio will be downmixed first.
                Otherwise, expect `x` to be a multichannel waveform tensor with
                shape `(n_channels, n_samples)`.

        Returns:
            Tuple[Tensor, Tensor]: a tuple of tensors, where the first
                tensor contains the output class probabilities
                (shape `(n_timesteps, n_labels)`), and the second tensor contains
                timestamps with start and end times for each label,
                shape `(n_timesteps, 2)`.
        """
        raise NotImplementedError("implement me!")
