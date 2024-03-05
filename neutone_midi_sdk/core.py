import torch as tr
from torch import nn, Tensor
from typing import List, Dict, Tuple, Union
from abc import abstractmethod
from neutone_midi_sdk.tokenization import TokenData
from neutone_midi_sdk.parameter import NeutoneParameter
import neutone_midi_sdk.constants as constants


class NeutoneMIDIModel(tr.nn.Module):
    def __init__(self,
                 model: tr.nn.Module,
                 vocab: Dict[str, int],
                 tokenizer_type: str,
                 tokenizer_data: TokenData):

        super().__init__()
        self.MAX_N_PARAMS = constants.MAX_N_PARAMS
        self.SDK_VERSION = constants.SDK_VERSION

        self.n_neutone_parameters = len(self.get_neutone_parameters())

        # Ensure number of parameters is within the maximum allowed
        assert self.n_neutone_parameters <= self.MAX_N_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )

        # Save parameter metadata
        self.neutone_parameters_metadata = {
            f"p{idx + 1}": p.to_metadata_dict()
            for idx, p in enumerate(self.get_neutone_parameters())
        }

        # Allocate default params buffer to prevent dynamic allocations later
        numerical_default_param_vals = self._get_numerical_default_param_values()
        default_param_values_t = tr.tensor([v for _, v in numerical_default_param_vals])
        assert default_param_values_t.size(0) <= self.MAX_N_PARAMS, (
            f"Number of default parameter values ({default_param_values_t.size(0)}) "
            f"exceeds the maximum allowed ({self.MAX_N_PARAMS})."
        )
        default_param_values = default_param_values_t.unsqueeze(-1)
        self.register_buffer("default_param_values", default_param_values)

        # Allocate remapped params dictionary to prevent dynamic allocations later
        self.remapped_params = {
            name: tr.tensor([val])
            for name, val in numerical_default_param_vals
        }

        # Save parameter information
        self.neutone_parameter_names = [p.name for p in self.get_neutone_parameters()]
        # TODO(cm): remove from here once plugin metadata parsing is implemented
        self.neutone_parameter_descriptions = [
            p.description for p in self.get_neutone_parameters()
        ]
        self.neutone_parameter_used = [p.used for p in self.get_neutone_parameters()]
        self.neutone_parameter_types = [
            p.type.value for p in self.get_neutone_parameters()
        ]

        ####################################
        # Old constructor
        # TODO(nic): remove when migrating to main Neutone core infra

        # parameters
        # neutone_parameters = self.get_neutone_parameters() # now handled by _get_numerical_default_param_values()
        # if len(neutone_parameters) < self.MAX_N_PARAMS:    # now handled in MidiToMidiBase 
        #     neutone_parameters += [
        #                               NeutoneParameter(
        #                                   name="",
        #                                   description="",
        #                                   used=False,
        #                                   default_value=0.0,
        #                               )
        #                            ] * (self.MAX_N_PARAMS - len(neutone_parameters))

        # default_param_values = tr.tensor(               # now handled in default_param_values_t
        #     [
        #         neutone_parameter.default_value
        #         for neutone_parameter in neutone_parameters
        #     ]
        # ).unsqueeze(-1)
        # self.register_buffer("default_param_values", default_param_values)
        # self.remapped_params = {
        #     neutone_param.name: default_param_values[idx]
        #     for idx, neutone_param in enumerate(self.get_neutone_parameters())
        # }

        ####################################

        # instantiate model
        model.eval()
        self.model = model

        # Setup tokenization methods
        assert tokenizer_type in constants.SUPPORTED_TOKENIZATIONS, \
            f"{tokenizer_type} not a recognized tokenization format."
        tokenizer_data = generate_fake_token_data() if tokenizer_data is None else tokenizer_data
        vocab = {"v": 0} if vocab is None else vocab
        self.midi_to_token_vocab = vocab
        self.token_to_midi_vocab = {v: k for k, v in vocab.items()}
        self.tokenizer_type = tokenizer_type
        self.tokenizer_data: TokenData = TokenData(tokenizer_data.strings, tokenizer_data.floats, tokenizer_data.ints)

    @abstractmethod
    def _get_numerical_default_param_values(
        self,
    ) -> List[Tuple[str, Union[float, int]]]:
        """
        Returns a list of tuples containing the name and default value of each
        numerical (float or int) parameter.
        This should not be overwritten by SDK users.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Set the model name
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

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return []
    
    @tr.jit.export
    def get_neutone_parameters_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Returns the metadata of the parameters as a string dictionary of string
        dictionaries.
        """
        return self.neutone_parameters_metadata

    @tr.jit.export
    def get_default_param_values(self) -> Tensor:
        """
        Returns the default parameter values as a tensor of shape
        (N_DEFAULT_PARAM_VALUES, 1).
        """
        return self.default_param_values
    
    @tr.jit.export
    def get_default_param_names(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_names

    @tr.jit.export
    def get_default_param_descriptions(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_descriptions

    @tr.jit.export
    def get_default_param_types(self) -> List[str]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_types

    @tr.jit.export
    def get_default_param_used(self) -> List[bool]:
        # TODO(nic): remove this once plugin metadata parsing is implemented
        return self.neutone_parameter_used

    def prepare_for_inference(self) -> None:
        self.model.eval()
        self.eval()


# Todo: Would like to deprecate this method, it is used in "HVO" format where there is no TokenData necessary
def generate_fake_token_data():
    token_strings: Dict[str, List[str]] = {"value": ["value"]}
    token_floats: Dict[str, List[float]] = {"value": [0.0]}
    token_ints: Dict[str, List[int]] = {"value": [0]}
    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)
    return token_data
