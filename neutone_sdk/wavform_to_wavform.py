import logging
import os
from abc import abstractmethod
from typing import Dict, List, Optional, Any, Final

import torch as tr
from torch import nn, Tensor

from neutone_sdk import NeutoneModel
from neutone_sdk.utils import validate_waveform

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class WaveformToWaveformBase(NeutoneModel):
    input_mono: Final[bool]
    output_mono: Final[bool]
    native_sample_rates: Final[List[int]]
    native_buffer_sizes: Final[List[int]]
    min_buffer_size: Final[Optional[int]]
    max_buffer_size: Final[Optional[int]]

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        self.input_mono = self.is_input_mono()
        self.output_mono = self.is_output_mono()
        self.native_sample_rates = self.get_native_sample_rates()
        self.native_buffer_sizes = self.get_native_buffer_sizes()
        if self.native_buffer_sizes:
            self.min_buffer_size = min(self.native_buffer_sizes)
            self.max_buffer_size = max(self.native_buffer_sizes)
        else:
            self.min_buffer_size = None
            self.max_buffer_size = None

    @abstractmethod
    def is_input_mono(self) -> bool:
        pass

    @abstractmethod
    def is_output_mono(self) -> bool:
        pass

    @abstractmethod
    def get_native_sample_rates(self) -> List[int]:
        """
        Returns a list of sample rates that the model was developed and tested
        with. If the list is empty, all common sample rates are assumed to be
        supported.
        Example value: [44100, 48000]
        """
        pass

    @abstractmethod
    def get_native_buffer_sizes(self) -> List[int]:
        """
        Returns a list of buffer sizes that the model was developed and tested
        with. If the list is empty, all common buffer sizes are assumed to be
        supported.
        Example value: [512, 1024, 2048]
        """
        pass

    @abstractmethod
    def do_forward_pass(
        self, x: Tensor, params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        Perform a forward pass on a waveform-to-waveform model.
        TODO(christhetree)
        """
        pass

    def forward(self, x: Tensor, params: Optional[Tensor] = None) -> Tensor:
        """
        Internal forward pass for a WaveformToWaveform model.
        TODO(christhetree)
        """
        validate_waveform(x)
        if params is None:
            x = self.do_forward_pass(x)
        else:
            remapped_params = {
                param.name: params[idx]
                for idx, param in enumerate(self.get_parameters())
            }
            x = self.do_forward_pass(x, remapped_params)

        validate_waveform(x)
        return x

    @tr.jit.export
    def calc_min_delay_samples(self) -> int:
        """
        If the model introduces a minimum amount of delay to the output audio,
        for example due to a lookahead buffer or cross-fading, return it here
        so that it can be forwarded to the DAW to compensate. Defaults to 0.

        This value may change if set_buffer_size() is used to change the buffer
        size, hence this is not a constant attribute.
        """
        return 0

    @tr.jit.export
    def set_buffer_size(self, n_samples: int) -> bool:
        """
        If the model supports dynamic buffer size resizing, add the
        functionality here.

        Args:
            n_samples: The number of samples to resize the buffer to.

        Returns:
            bool: True if successful, False if not supported or unsuccessful.
                  Defaults to False.
        """
        return False

    @tr.jit.export
    def flush(self) -> Optional[Tensor]:
        """
        If the model supports flushing (e.g. due to a delay from a lookahead
        buffer or cross-fading etc.) add the functionality here.

        Returns:
            Optional[Tensor]: None if not supported, otherwise a right side zero
                              padded tensor of length buffer_size with the
                              flushed samples at the beginning.
        """
        return None

    @tr.jit.export
    def reset(self) -> bool:
        """
        If the model supports resetting (e.g. wiping internal state), add the
        functionality here.

        Returns:
            bool: True if successful, False if not supported or unsuccessful.
                  Defaults to False.
        """
        return False

    def get_preserved_attributes(self) -> List[str]:
        preserved_attrs = super().get_preserved_attributes()
        preserved_attrs.extend(
            [
                self.calc_min_delay_samples.__name__,
                self.set_buffer_size.__name__,
                self.flush.__name__,
                self.reset.__name__,
            ]
        )
        return preserved_attrs

    def to_metadata_dict(self) -> Dict[str, Any]:
        metadata_dict = super().to_metadata_dict()
        metadata_dict["input_mono"] = self.input_mono
        metadata_dict["output_mono"] = self.output_mono
        metadata_dict["native_sample_rates"] = self.native_sample_rates
        metadata_dict["native_buffer_sizes"] = self.native_buffer_sizes
        return metadata_dict
