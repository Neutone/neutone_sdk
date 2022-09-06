import logging
from abc import abstractmethod
from typing import NamedTuple, Dict, List, Optional

import torch as tr
from torch import Tensor

from neutone_sdk import NeutoneModel
from neutone_sdk.utils import validate_waveform

logging.basicConfig()
log = logging.getLogger(__name__)


class WaveformToWaveformMetadata(NamedTuple):
    model_name: str
    model_authors: List[str]
    model_version: str
    model_short_description: str
    model_long_description: str
    technical_description: str
    technical_links: Dict[str, str]
    tags: List[str]
    citation: str
    is_experimental: bool
    neutone_parameters: Dict[str, Dict[str, str]]
    wet_default_value: float
    dry_default_value: float
    input_gain_default_value: float
    output_gain_default_value: float
    is_input_mono: bool
    is_output_mono: bool
    model_type: str
    native_sample_rates: List[int]
    native_buffer_sizes: List[int]
    sdk_version: str
    pytorch_version: str


class WaveformToWaveformBase(NeutoneModel):
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
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """
        SDK users can overwrite this method to implement the logic for their models.

        Args:
            x:
                torch Tensor of shape [num_channels, num_samples]
                num_channels is 1 if `is_input_mono` is set to True, otherwise 2
                num_samples will be one of the sizes specificed in native_buffer_sizes

                The sample rate of the audio will also be one of the ones specific in
                native_sample_rates.

                The best combination is chosen based on the DAW parameters at runtime. If
                unsure, provide only one value in the lists.
            params:
                Python dictionary mapping from parameter names (defined by the values in
                get_parameters) to values. By default, we aggregate the sample values over the
                entire buffer and provide the mean value.

                Override the `remap_params_for_forward_pass` method for more fine grained control.

        Returns:
            torch Tensor of shape [num_channels, num_samples]

            The shape of the output must match the shape of the input.
        """
        pass

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

    def aggregate_param(self, param: Tensor) -> Tensor:
        """
        Aggregates a parameter of size (buffer_size,) to a single value.

        By default we take the mean value of to provide a single value
        for the current buffer.

        For more fine grained control, override this method as required.
        """
        if self.use_debug_mode:
            assert param.ndim == 1
        agg_param = tr.mean(
            param, dim=0, keepdim=True
        )  # TODO(christhetree): prevent memory allocation
        return agg_param

    def forward(self, x: Tensor, params: Optional[Tensor] = None) -> Tensor:
        """
        Internal forward pass for a WaveformToWaveform model.

        If params is None, we fill in the default values.
        """
        if params is None:
            # The default params come in as one value by default but for compatibility
            # with the plugin inputs we repeat them for the size of the buffer
            # TODO(christhetree): try expand instead of repeat to avoid memory allocation
            # params = self.get_default_param_values().expand(-1, x.shape[1])
            params = self.get_default_param_values().repeat(1, x.shape[1])

        if self.use_debug_mode:
            assert params.shape == (self.MAX_N_PARAMS, x.shape[1])
            validate_waveform(x, self.is_input_mono())

        remapped_params = {
            param.name: self.aggregate_param(value)
            for param, value in zip(self.get_neutone_parameters(), params)
        }
        x = self.do_forward_pass(x, remapped_params)

        if self.use_debug_mode:
            validate_waveform(x, self.is_output_mono())

        return x

    @tr.jit.export
    def is_resampling(self) -> bool:
        # w2w wrapper does not support resampling by default
        return False

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
    def set_daw_sample_rate_and_buffer_size(
        self,
        daw_sr: int,
        daw_buffer_size: int,
        model_sr: Optional[int] = None,
        model_buffer_size: Optional[int] = None,
    ) -> None:
        # w2w only supports changing buffer size
        self.set_buffer_size(daw_buffer_size)

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

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        # This avoids using inheritance which torchscript does not support
        preserved_attrs = self.get_core_preserved_attributes()
        preserved_attrs.extend(
            [
                "is_input_mono",
                "is_output_mono",
                "get_native_sample_rates",
                "get_native_buffer_sizes",
                "is_resampling",
                "calc_min_delay_samples",
                "set_daw_sample_rate_and_buffer_size",
                "reset",
                "get_preserved_attributes",
                "to_metadata",
            ]
        )
        return preserved_attrs

    @tr.jit.export
    def to_metadata(self) -> WaveformToWaveformMetadata:
        # This avoids using inheritance which torchscript does not support
        core_metadata = self.to_core_metadata()
        return WaveformToWaveformMetadata(
            model_name=core_metadata.model_name,
            model_authors=core_metadata.model_authors,
            model_short_description=core_metadata.model_short_description,
            model_long_description=core_metadata.model_long_description,
            neutone_parameters=core_metadata.neutone_parameters,
            wet_default_value=core_metadata.wet_default_value,
            dry_default_value=core_metadata.dry_default_value,
            input_gain_default_value=core_metadata.input_gain_default_value,
            output_gain_default_value=core_metadata.output_gain_default_value,
            technical_description=core_metadata.technical_description,
            technical_links=core_metadata.technical_links,
            tags=core_metadata.tags,
            model_version=core_metadata.model_version,
            sdk_version=core_metadata.sdk_version,
            pytorch_version=core_metadata.pytorch_version,
            citation=core_metadata.citation,
            is_experimental=core_metadata.is_experimental,
            is_input_mono=self.is_input_mono(),
            is_output_mono=self.is_output_mono(),
            model_type=f"{'mono' if self.is_input_mono() else 'stereo'}-{'mono' if self.is_output_mono() else 'stereo'}",
            native_buffer_sizes=self.get_native_buffer_sizes(),
            native_sample_rates=self.get_native_sample_rates(),
        )
