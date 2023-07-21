import torch
from typing import List, Dict
from abc import abstractmethod
from neutone_midi_sdk.tokenization import TokenData
from neutone_midi_sdk.parameter import NeutoneParameter
import neutone_midi_sdk.constants as constants


class NeutoneMIDIModel(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 vocab: Dict[str, int],
                 tokenizer_type: str,
                 tokenizer_data: TokenData):

        super().__init__()
        self.MAX_N_PARAMS = constants.MAX_N_PARAMS
        self.SDK_VERSION = constants.SDK_VERSION

        assert len(self.get_neutone_parameters()) <= self.MAX_N_PARAMS
        # Ensure parameter names are unique
        assert len(set([p.name for p in self.get_neutone_parameters()])) == len(
            self.get_neutone_parameters()
        )

        # instantiate model
        model.eval()
        self.model = model

        # parameters
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

        default_param_values = torch.tensor(
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

    def get_default_param_values(self) -> torch.Tensor:
        return self.default_param_values

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
