import torch
from typing import List, Dict, Optional
from abc import abstractmethod
from neutone_midi_sdk import NeutoneMIDIModel, convert_midi_to_tokens, convert_tokens_to_midi, TokenData



class MidiToMidiBase(NeutoneMIDIModel):
    def __init__(self,
                 model: torch.nn.Module,
                 vocab: Dict[str, int],
                 tokenizer_type: str,
                 tokenizer_data: TokenData,
                 add_dimension: bool = True):
        super().__init__(model, vocab, tokenizer_type, tokenizer_data)
        self.add_dimension = add_dimension



    def prepare_for_inference(self) -> None:
        super().prepare_for_inference()

    @abstractmethod
    def do_forward_pass(self, tokenized_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        SDK users can overwrite this method to implement the logic of their models.
        The input is a tensor of data that has been tokenized according to the tokenization settings,
        i.e. REMI, TSD, etc.

        In addition to the forward pass of your model, you can incorporate additional logic, such as
        the control parameters.

        The model should return data in the same format it was input, i.e. REMI-in, REMI-out. This will then
        be de-tokenized in the top-level 'forward' method.
        """
        pass

    def forward(self, midi_data: torch.Tensor, params: Optional[torch.Tensor] = None) -> torch.Tensor:
        #in_n = midi_data.size(1)
        if params is None:
            params = self.get_default_param_values()#.repeat(1, in_n)

        for idx, neutone_param in enumerate(self.get_neutone_parameters()):
            self.remapped_params[neutone_param.name] = params[idx]

        tokenized_data = convert_midi_to_tokens(midi_data=midi_data,
                                                token_type=self.tokenizer_type,
                                                midi_to_token_vocab=self.midi_to_token_vocab,
                                                tokenizer_data=self.tokenizer_data)


        if self.add_dimension:
            tokenized_data = torch.unsqueeze(tokenized_data, dim=0)
        model_output = self.do_forward_pass(tokenized_data, self.remapped_params)
        if self.add_dimension:
            model_output = torch.squeeze(model_output, dim=0)

        output_midi_data = convert_tokens_to_midi(tokens=model_output,
                                                  token_type=self.tokenizer_type,
                                                  token_to_midi_vocab=self.token_to_midi_vocab,
                                                  tokenizer_data=self.tokenizer_data)

        return output_midi_data


def generate_fake_token_data():
    token_strings: Dict[str, List[str]] = {"value": ["value"]}
    token_floats: Dict[str, List[float]] = {"value": [0.0]}
    token_ints: Dict[str, List[int]] = {"value": [0]}
    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)
    return token_data
