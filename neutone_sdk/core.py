import logging
import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List, Final

import torch as tr
from torch import nn, Tensor

from neutone_sdk import constants
from neutone_sdk.parameter import NeutoneParameter

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class CoreMetadata(NamedTuple):
    model_name: str
    model_authors: List[str]
    model_short_description: str
    model_long_description: str
    technical_description: str
    technical_links: Dict[str, str]
    neutone_parameters: Dict[str, Dict[str, str]]
    wet_default_value: float
    dry_default_value: float
    output_gain_default_value: float
    tags: List[str]
    model_version: str
    sdk_version: str
    citation: str
    is_experimental: bool


class NeutoneModel(ABC, nn.Module):
    MAX_N_PARAMS: Final[int] = 4
    SDK_VERSION: Final[str] = constants.SDK_VERSION

    # TODO(christhetree): check all preserved_attrs have been exported
    def __init__(self, model: nn.Module) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()
        self.MAX_N_PARAMS = NeutoneModel.MAX_N_PARAMS
        self.SDK_VERSION = NeutoneModel.SDK_VERSION

        assert len(self.get_parameters()) <= self.MAX_N_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_parameters()])) == len(
            self.get_parameters()
        )
        model.eval()
        self.model = model

        # Convert parameters to metadata format
        parameters = self.get_parameters()
        if len(parameters) < self.MAX_N_PARAMS:
            parameters += [
                NeutoneParameter(
                    name="",
                    description="",
                    used=False,
                    default_value=0.0,
                )
            ] * (self.MAX_N_PARAMS - len(parameters))
        self.parameters_metadata = {
            f"p{idx + 1}": param.to_metadata_dict()
            for idx, param in enumerate(parameters)
        }

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
    def get_model_version(self) -> str:
        pass

    @abstractmethod
    def is_experimental(self) -> bool:
        pass

    def get_technical_links(self) -> Dict[str, str]:
        return {}

    def get_citation(self) -> str:
        return ""

    def get_parameters(self) -> List[NeutoneParameter]:
        return []

    def get_default_parameters(self) -> Tensor:
        return tr.tensor(
            [param.default_value for param in self.get_parameters()]
        ).reshape(-1, 1)

    def get_wet_default_value(self) -> float:
        return 1.0

    def get_dry_default_value(self) -> float:
        return 0.0

    def get_output_gain_default_value(self) -> float:
        return 0.5

    def get_preserved_attributes(self) -> List[str]:
        return [self.to_core_metadata.__name__, self.get_default_parameters.__name__]

    @tr.jit.export
    def to_core_metadata(self) -> CoreMetadata:
        return CoreMetadata(
            model_name=self.get_model_name(),
            model_authors=self.get_model_authors(),
            model_short_description=self.get_model_short_description(),
            model_long_description=self.get_model_long_description(),
            neutone_parameters=self.parameters_metadata,
            wet_default_value=self.get_wet_default_value(),
            dry_default_value=self.get_dry_default_value(),
            output_gain_default_value=self.get_output_gain_default_value(),
            technical_description=self.get_technical_description(),
            technical_links=self.get_technical_links(),
            tags=self.get_tags(),
            model_version=self.get_model_version(),
            sdk_version=self.SDK_VERSION,
            citation=self.get_citation(),
            is_experimental=self.is_experimental(),
        )
