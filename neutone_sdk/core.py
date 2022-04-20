import logging
import os
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List, Final, Union

import torch as tr
from torch import nn

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
    tags: List[str]
    version: int


class NeutoneModel(ABC, nn.Module):
    MAX_N_PARAMS: Final[int] = 4

    # TODO(christhetree): check all preserved_attrs have been exported
    def __init__(self, model: nn.Module) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()
        self.MAX_N_PARAMS = NeutoneModel.MAX_N_PARAMS
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
            parameters += [NeutoneParameter(
                name="",
                description="",
                used=False,
            )] * (self.MAX_N_PARAMS - len(parameters))
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
    def get_version(self) -> int:
        pass

    def get_technical_links(self) -> Dict[str, str]:
        return {}

    def get_parameters(self) -> List[NeutoneParameter]:
        return []

    def get_preserved_attributes(self) -> List[str]:
        return [self.to_core_metadata.__name__]

    @tr.jit.export
    def to_core_metadata(self) -> CoreMetadata:
        return CoreMetadata(
            model_name=self.get_model_name(),
            model_authors=self.get_model_authors(),
            model_short_description=self.get_model_short_description(),
            model_long_description=self.get_model_long_description(),
            neutone_parameters=self.parameters_metadata,
            technical_description=self.get_technical_description(),
            technical_links=self.get_technical_links(),
            tags=self.get_tags(),
            version=self.get_version(),
        )
