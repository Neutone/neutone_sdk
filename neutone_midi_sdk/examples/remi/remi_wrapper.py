import json
import torch
from typing import Dict, List
from neutone_midi_sdk import MidiToMidiBase, NeutoneParameter, TokenData, prepare_token_data


class RemiModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x


class RemiModelWrapper(MidiToMidiBase):
    """
    Here you can define overwrite several methods to define your model's functionality.
    This most important is "do_forward_pass"; this is where you can define custom behavior, such
    as inserting parameters or modifying the sampling logic.
    """
    def get_model_name(self) -> str:
        return "neutone_remi"

    def get_model_authors(self) -> List[str]:
        return ["Julian Lenz"]

    def get_model_short_description(self) -> str:
        return "REMI melody generation"

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("temperature", "sampling temp", default_value=0.6)
        ]

    def do_forward_pass(self, tokenized_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # in reality this model doesn't use params; but here is how you can retrieve it
        temperature = params["temperature"]
        output = self.model.forward(tokenized_data)
        return output


if __name__ == "__main__":

    # load config and vocab files
    with open("config.json", "r") as fp:
        config = json.load(fp)
    with open("vocab.json", "r") as fp:
        vocab = json.load(fp)

    # Get pre-processed data
    tokenizer_type = "REMI"
    tokenizer_data: TokenData = prepare_token_data(tokenizer_type, vocab, config)

    # Load model
    # Normally you would load a trained model; for this demo, we have the dummy model instead
    # scripted_model = torch.jit.load("path_to_trained_model.pt")
    scripted_model = RemiModel()

    # Wrap it with SDK and export
    wrapped_model = RemiModelWrapper(model=scripted_model,
                                     vocab=vocab,
                                     tokenizer_type=tokenizer_type,
                                     tokenizer_data=tokenizer_data)
    scripted_model = torch.jit.script(wrapped_model)
    scripted_model.save("neutone_remi_model.pt")


