import logging
import os
import time
from abc import ABC, abstractmethod
from typing import NamedTuple, Dict, List

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
    input_gain_default_value: float
    output_gain_default_value: float
    tags: List[str]
    model_version: str
    sdk_version: str
    date_created: float
    pytorch_version: str
    citation: str
    is_experimental: bool


class NeutoneModel(ABC, nn.Module):
    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()
        self.MAX_N_PARAMS = constants.MAX_N_PARAMS
        self.SDK_VERSION = constants.SDK_VERSION
        self.CURRENT_TIME = time.time()
        self.use_debug_mode = use_debug_mode

        assert len(self.get_neutone_parameters()) <= self.MAX_N_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )
        model.eval()
        self.model = model

        # Convert neutone_parameters to metadata format
        neutone_parameters = self.get_neutone_parameters()
        if len(neutone_parameters) < self.MAX_N_PARAMS:
            neutone_parameters += [
                NeutoneParameter(
                    name="",
                    description="",
                    used=False,
                    default_value=0.0,
                )
            ] * (self.MAX_N_PARAMS - len(neutone_parameters))
        self.neutone_parameters_metadata = {
            f"p{idx + 1}": neutone_parameter.to_metadata_dict()
            for idx, neutone_parameter in enumerate(neutone_parameters)
        }
        default_param_values = tr.tensor(
            [
                neutone_parameter.default_value
                for neutone_parameter in neutone_parameters
            ]
        ).unsqueeze(-1)
        self.register_buffer("default_param_values", default_param_values)
        self.remapped_params = {
            neutone_param.name: default_param_values[idx]
            for idx, neutone_param in enumerate(self.get_neutone_parameters())
        }
        # This is required for TorchScript typing when there are no Neutone parameters defined
        self.remapped_params["__torchscript_typing"] = default_param_values[0]

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Used to set the model name. This will be displayed on both the
        website and the plugin.

        Maximum length of 30 characters.
        """
        pass

    @abstractmethod
    def get_model_authors(self) -> List[str]:
        """
        Used to set the model authors. This will be displayed on both the
        website and the plugin.

        Should reflect the name of the people that developed the wrapper
        of the model using the SDK. Can be different from the authors of
        the original model.

        Maximum of 5 authors.
        """
        pass

    @abstractmethod
    def get_model_short_description(self) -> str:
        """
        Used to set the model short description. This will be displayed on both
        the website and the plugin.

        This is meant to be seen by the audio creators and should give a summary
        of what the model does.

        Maximum of 150 characters.
        """
        pass

    @abstractmethod
    def get_model_long_description(self) -> str:
        """
        Used to set the model long description. This will be displayed only on
        the website.

        This is meant to be seen by the audio creators and should give an extensive
        description of what the model does. Could describe interesting uses of the
        model, good combinations of parameters, what types of audio has it been
        tested with etc.

        Maximum of 500 characters.
        """
        pass

    @abstractmethod
    def get_technical_description(self) -> str:
        """
        Used to set the model technical description. This will be displayed only on
        the website.

        This is meant to be seen by other researchers or people that want to develop
        similar models. It could present a summary of the internals of the model:
        what architecture it is based on, what kind of data it was trained with,
        on what kind of hardware.

        If the authors of the plugin are different from the authors of the model(s)
        included this section along with citation and technical links are places
        to provide appropiate credits.

        Maximum of 500 characters.
        """
        pass

    @abstractmethod
    def get_tags(self) -> List[str]:
        """
        Used to provide a list of tags. This will be displayed on the website and will
        be used later on for filtering of similar models.

        Maximum of 7 tags of 15 characters each.
        """
        pass

    @abstractmethod
    def get_model_version(self) -> str:
        """
        Used to set the model version. This will be displayed on both the website and the plugin.

        We suggest people use semantic versioning for their models, but in a lot of cases it can
        be overkill. For now we only support showing the latest version of the model.

        Please provide a string like "1", "1.0", "1.0.0", "0.1.0" etc.
        """
        pass

    @abstractmethod
    def is_experimental(self) -> bool:
        """
        Used to set the experimental flag. This will be displayed on both the website and the plugin.

        If this flag is set the models will have a special icon next to them signaling to the users of
        the plugin that this model is an experimental release.
        """
        pass

    def get_technical_links(self) -> Dict[str, str]:
        """
        Used to set the hechnical links. These will be displayed only on the website.

        Under the technical description field the following links can be displayed as buttons.
        This can be used to provide links to the implementation, to scientific paper, personal websites etc.

        While any key-value pair can be provided, we strongly encourage users to provide a dictionary
        with keys such as Paper, Code, Personal, GitHub, Blog, Twitter, Instagram etc.

        Maximum of 3 links.
        """
        return {}

    def get_citation(self) -> str:
        """
        Used to set the citation. This will be displayed only on the website.

        This field is specifically meant to display the citation for a scientific paper that the model
        is based on, if any. Will be displayed under the technical links. Can be left empty.

        Maximum of 150 characters.
        """
        return ""

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []

    def get_default_param_values(self) -> Tensor:
        return self.default_param_values

    def get_wet_default_value(self) -> float:
        return 1.0

    def get_dry_default_value(self) -> float:
        return 0.0

    def get_input_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    def get_output_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    def prepare_for_inference(self) -> None:
        """Prepare a model for inference and to be converted to torchscript."""
        self.use_debug_mode = False
        self.model.eval()
        self.eval()

    @tr.jit.export
    def get_core_preserved_attributes(self) -> List[str]:
        return [
            "to_core_metadata",
            "model",
            "default_param_values",
            "get_core_preserved_attributes",
        ]

    @tr.jit.export
    def to_core_metadata(self) -> CoreMetadata:
        return CoreMetadata(
            model_name=self.get_model_name(),
            model_authors=self.get_model_authors(),
            model_short_description=self.get_model_short_description(),
            model_long_description=self.get_model_long_description(),
            neutone_parameters=self.neutone_parameters_metadata,
            wet_default_value=self.get_wet_default_value(),
            dry_default_value=self.get_dry_default_value(),
            input_gain_default_value=self.get_input_gain_default_value(),
            output_gain_default_value=self.get_output_gain_default_value(),
            technical_description=self.get_technical_description(),
            technical_links=self.get_technical_links(),
            tags=self.get_tags(),
            model_version=self.get_model_version(),
            sdk_version=self.SDK_VERSION,
            pytorch_version=tr.__version__,
            date_created=self.CURRENT_TIME,
            citation=self.get_citation(),
            is_experimental=self.is_experimental(),
        )
