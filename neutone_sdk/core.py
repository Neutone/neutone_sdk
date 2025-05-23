import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any

import torch as tr
from torch import nn, Tensor

from neutone_sdk import constants
from neutone_sdk.parameter import NeutoneParameter

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NeutoneModel(ABC, nn.Module):
    # TorchScript typing does not support instance attributes, so we need to type them
    # as class attributes. This is required for supporting models with no parameters.
    # (https://github.com/pytorch/pytorch/issues/51041#issuecomment-767061194)
    neutone_parameters_metadata: Dict[
        str, Dict[str, Union[int, float, str, bool, List[str], List[int]]]
    ]
    neutone_parameter_names: List[str]

    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        """
        Creates an Neutone model, wrapping a child model (that does the real
        work).
        """
        super().__init__()

        # Save and prepare model. This should be done at the very beginning of the
        # constructor to enable accessing the model in other methods of this class.
        model.eval()
        self.model = model

        self.MAX_N_PARAMS = self._get_max_n_params()
        self.SDK_VERSION = constants.SDK_VERSION
        self.CURRENT_TIME = time.time()
        self.use_debug_mode = use_debug_mode
        self.n_neutone_parameters = len(self.get_neutone_parameters())

        # Ensure the number of parameters is within the allowed limit
        assert self.n_neutone_parameters <= self.MAX_N_PARAMS, (
            f"Number of parameters ({self.n_neutone_parameters}) exceeds the maximum "
            f"allowed ({self.MAX_N_PARAMS})."
        )
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )

        # Save parameter metadata
        self.neutone_parameters_metadata = {
            f"p{idx + 1}": p.to_metadata()
            for idx, p in enumerate(self.get_neutone_parameters())
        }

        # Allocate default params buffer to prevent dynamic allocations later
        default_vals_0to1 = self._get_numerical_params_default_values_0to1()
        n_numerical_params = default_vals_0to1.size(0)
        assert n_numerical_params <= self.MAX_N_PARAMS, (
            f"Number of default param values ({n_numerical_params}) "
            f"exceeds the maximum allowed ({self.MAX_N_PARAMS})."
        )
        default_vals_0to1 = default_vals_0to1.view(n_numerical_params, 1)
        self.register_buffer("numerical_params_default_values_0to1", default_vals_0to1)

        # Save parameter information
        self.neutone_parameter_names = [p.name for p in self.get_neutone_parameters()]

    @abstractmethod
    def _get_max_n_params(self) -> int:
        """
        Sets the maximum number of parameters that the model can have.
        This should not be overwritten by SDK users.
        """
        pass

    @abstractmethod
    def _get_numerical_params_default_values_0to1(
        self,
    ) -> Tensor:
        """
        Returns a float tensor with the default values of the numerical parameters
        in the range [0, 1].
        This should not be overwritten by SDK users.
        """
        pass

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

    def prepare_for_inference(self) -> None:
        """Prepare a model for inference and to be converted to torchscript."""
        self.use_debug_mode = False
        self.model.eval()
        self.eval()

    @tr.jit.export
    def get_neutone_parameters_metadata(
        self,
    ) -> Dict[str, Dict[str, Union[int, float, str, bool, List[str], List[int]]]]:
        """
        Returns the metadata of the parameters as a dictionary of ParameterMetadata
        named tuples.
        """
        return self.neutone_parameters_metadata

    @tr.jit.export
    def get_numerical_params_default_values_0to1(self) -> Tensor:
        """
        Returns the default parameter values as a tensor of shape
        (n_numerical_params, 1).
        """
        return self.numerical_params_default_values_0to1

    @tr.jit.export
    def get_wet_default_value(self) -> float:
        return 1.0

    @tr.jit.export
    def get_dry_default_value(self) -> float:
        return 0.0

    @tr.jit.export
    def get_input_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    @tr.jit.export
    def get_output_gain_default_value(self) -> float:
        """[0.0, 1.0] here maps to [-30.0db, +30.0db]"""
        return 0.5

    @tr.jit.export
    def get_core_preserved_attributes(self) -> List[str]:
        return [
            "model",  # nn.Module
            "get_neutone_parameters_metadata",
            "get_numerical_params_default_values_0to1",
            "get_wet_default_value",
            "get_dry_default_value",
            "get_input_gain_default_value",
            "get_output_gain_default_value",
            "get_core_preserved_attributes",
            "to_core_metadata",
        ]

    @tr.jit.export
    def to_core_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.get_model_name(),
            "model_authors": self.get_model_authors(),
            "model_short_description": self.get_model_short_description(),
            "model_long_description": self.get_model_long_description(),
            "neutone_parameters": self.get_neutone_parameters_metadata(),
            "wet_default_value": self.get_wet_default_value(),
            "dry_default_value": self.get_dry_default_value(),
            "input_gain_default_value": self.get_input_gain_default_value(),
            "output_gain_default_value": self.get_output_gain_default_value(),
            "technical_description": self.get_technical_description(),
            "technical_links": self.get_technical_links(),
            "tags": self.get_tags(),
            "model_version": self.get_model_version(),
            "sdk_version": self.SDK_VERSION,
            "pytorch_version": tr.__version__,
            "date_created": self.CURRENT_TIME,
            "citation": self.get_citation(),
            "is_experimental": self.is_experimental(),
        }
