import logging
import argparse
import os
import pathlib
from argparse import ArgumentParser
from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from neutone_sdk.gcn_1d import GCN1D

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def initialize_model(device, config):
    """
    Initialize a model based on the model type specified in the hparams.

    Parameters:
        device: torch.device
            The device (e.g., 'cuda' or 'cpu') to which the model should be transferred.
        hparams: dict
            Hyperparameters dictionary containing settings and specifications for the model.

    Returns:
        model: torch.nn.Module
            nitialized model of the type specified in hparams.
        rf: int or None
            Receptive field of the model in terms of samples.
            Only computed for specific model types ["TCN", "PedalNetWaveNet", "GCN"].
            Returns None for other model types.
        params: int
            Total number of trainable parameters in the model.
    """
    model_dict = {
        "GCN": GCN,
    }

    model_params = {
        "GCN": {
            "in_ch",
            "out_ch",
            "n_blocks",
            "n_channels",
            "dilation_growth",
            "kernel_size",
            "cond_dim",
        },
    }

    if config["model_type"] not in model_dict:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    # Filter hparams to only include the keys specific to the model type
    filtered_hparams = {
        k: v for k, v in config.items() if k in model_params[config["model_type"]]
    }

    model = model_dict[config["model_type"]](**filtered_hparams).to(device)
    print(f"Configuration name: {config['name']}")

    rf = model.calc_receptive_field()
    print(
        f"Receptive field: {rf} samples or {(rf / config['sample_rate'])*1e3:0.1f} ms"
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")

    return model, rf, params


def load_model_checkpoint(checkpoint_path, device, args):
    """
    Load a model checkpoint from a given path.

    Parameters:
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') where the checkpoint will be loaded.
        checkpoint_path : str
            Path to the checkpoint file to be loaded.
        args :
            Additional arguments or configurations (currently unused in the function but can be
            utilized for future extensions).

    Returns:
        model : torch.nn.Module
            Initialized and state-loaded model from the checkpoint.
        optimizer_state_dict : dict or None
            State dictionary for the optimizer if present in the checkpoint; None otherwise.
        scheduler_state_dict : dict or None
            State dictionary for the learning rate scheduler if present in the checkpoint; None otherwise.
        hparams : dict
            Hyperparameters dictionary loaded from the checkpoint.
        rf : int or None
            Receptive field of the model in terms of samples, computed during model initialization.
            Only computed for specific model types; None for others.
        params : int
            Total number of trainable parameters in the model.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get("model_state_dict")
    optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
    scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
    loaded_config = checkpoint["config_state_dict"]

    model, rf, params = initialize_model(device, loaded_config)
    model.load_state_dict(model_state_dict)

    return model, optimizer_state_dict, scheduler_state_dict, loaded_config, rf, params


# TODO(christhetree): integrate this into tcn_1d.py
class PaddingCached(nn.Module):  # to maintain signal continuity over sample windows
    def __init__(self, padding: int, channels: int) -> None:
        super().__init__()
        self.padding = padding
        self.channels = channels
        pad = torch.zeros(1, self.channels, self.padding)
        self.register_buffer("pad", pad)

    def forward(self, x: Tensor) -> Tensor:
        padded_x = torch.cat([self.pad, x], -1)  # concat input signal to the cache
        self.pad = padded_x[..., -self.padding :]  # discard old cache
        return padded_x


# TODO(christhetree): integrate this into tcn_1d.py
class Conv1dCached(nn.Module):  # Conv1d with cache
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
        kernel: int,
        stride: int,
        padding: int,
        dilation: int = 1,
        weight_norm: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.pad = PaddingCached(padding * 2, in_chan)
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel, stride, dilation=dilation, bias=bias
        )
        nn.init.normal_(self.conv.weight)  # random initialization
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        return x


def replace_modules(module):
    for name, child in module.named_children():
        if isinstance(child, Conv1dCausal):
            # Create a new instance of Conv1dCached using the Conv1dCausal instance
            cached_conv = Conv1dCached(child)
            # Replace the Conv1dCausal instance with the Conv1dCached instance
            setattr(module, name, cached_conv)
        else:
            # If the child is not a Conv1dCausal instance, call the function recursively
            replace_modules(child)


class GCNModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "GCN.example"  # <- EDIT THIS

    def get_model_authors(self) -> List[str]:
        return ["Francesco Papaleo"]  # <- EDIT THIS

    def get_model_short_description(self) -> str:
        return "Neural spring reverb effect"  # <- EDIT THIS

    def get_model_long_description(self) -> str:
        return """
            Neural spring reverb effect through Gated Convolutional Neural Network with FiLM.
            Trained on EGFx dataset.
        """  # <- EDIT THIS

    def get_technical_description(self) -> str:
        return "GCN model based on the idea proposed by Comunità et al."  # <- EDIT THIS

    def get_tags(self) -> List[str]:
        return ["audio effect", "spring reverb", "GCN"]  # <- EDIT THIS

    def get_model_version(self) -> str:
        return "0.1.0"  # <- EDIT THIS

    def is_experimental(self) -> bool:
        return False  # <- EDIT THIS

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "http://arxiv.org/abs/2211.00497.pdf",
            "Code": "https://github.com/mcomunita/gcn-tfilm",
        }  # <- EDIT THIS

    def get_citation(self) -> str:
        return """Comunità, M., Steinmetz, C. J., Phan, H., & Reiss, J. D. (2023). 
        Modelling Black-Box Audio Effects with Time-Varying Feature Modulation. 
        https://doi.org/10.1109/icassp49357.2023.10097173"""  # <- EDIT THIS

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("depth", "Modulation Depth", 0.5),
            NeutoneParameter("FiLM1", "Feature modulation 1", 0.0),
            NeutoneParameter("FiLM2", "Feature modulation 2", 0.0),
            NeutoneParameter("FiLM3", "Feature modulation 3", 0.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True  # Input is mono

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True  # Output is mono

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # Set to model sample rate during training

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [2048]

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> torch.Tensor:
        # conditioning for FiLM layer
        p1 = params["FiLM1"]
        p2 = params["FiLM2"]
        p3 = params["FiLM3"]
        depth = params["depth"]
        cond = torch.stack([p1, p2, p3], dim=1) * depth
        cond = cond.expand(x.shape[0], 3)

        # forward pass
        x = x.unsqueeze(1)
        x = self.model(x, cond)
        x = x.squeeze(1)
        return x


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input",
        type=str,
        help="The path to the pytorch checkpoint file.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError("Checkpoint file not found")

    model, _, _, config, rf, params = load_model_checkpoint(
        checkpoint_path=args.input,
        device=torch.device("cpu"),
        args=None,
    )

    # Replace all instances of Conv1dCausal with Conv1dCached
    replace_modules(model)

    model_name = config["name"]
    DESTINATION_DIR = Path(f"neutone_models/{model_name}")

    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)

    # Export model to Neutone
    model = torch.jit.script(model.to("cpu"))
    model_wrapper = GCNModelWrapper(model)

    # Call the export function
    save_neutone_model(
        model=model_wrapper,
        root_dir=DESTINATION_DIR,
        dump_samples=True,
        submission=True,
        max_n_samples=3,
        freeze=False,
        optimize=False,
        speed_benchmark=True,
    )


if __name__ == "__main__":
    main()
