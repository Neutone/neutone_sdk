import logging
import os
from abc import abstractmethod
from typing import NamedTuple, Dict, List, Optional, Tuple, Union

import torch as tr
from torch import Tensor, nn

from neutone_sdk import NeutoneModel, constants, NeutoneParameterType
from neutone_sdk.utils import validate_waveform

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class NonRealtimeMetadata(NamedTuple):
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
    audio_in_channels: List[int]
    audio_out_channels: List[int]
    native_sample_rates: List[int]
    native_buffer_sizes: List[int]
    is_one_shot_model: bool
    is_text_model: bool
    audio_in_labels: List[str]
    audio_out_labels: List[str]
    sdk_version: str
    pytorch_version: str
    date_created: float


class NonRealtimeBase(NeutoneModel):
    ALLOWED_PARAM_TYPES = {
        NeutoneParameterType.CONTINUOUS,
        NeutoneParameterType.CATEGORICAL,
        NeutoneParameterType.TEXT,
    }
    # TorchScript typing does not support instance attributes, so we need to type them
    # as class attributes. This is required for supporting models with no parameters.
    # (https://github.com/pytorch/pytorch/issues/51041#issuecomment-767061194)
    cont_param_names: List[str]
    cont_param_indices: List[int]
    cat_param_names: List[str]
    cat_param_indices: List[int]
    cat_param_n_values: Dict[str, int]
    text_param_max_n_chars: List[int]
    text_param_default_values: List[str]

    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        """
        Wraps a PyTorch model for use in a non-realtime context.
        Compatible with the Neutone Gen plugin.
        """
        super().__init__(model, use_debug_mode)
        self.progress_percentage = 0
        self.cancel_forward_pass_requested = False
        self.has_text_param = False

        self.n_cont_params = 0
        self.cont_param_names = []
        self.cont_param_indices = []

        self.n_cat_params = 0
        self.cat_param_names = []
        self.cat_param_indices = []
        self.cat_param_n_values = {}

        self.n_text_params = 0
        self.text_param_max_n_chars = []
        self.text_param_default_values = []

        # We have to keep track of this manually since text params are separate
        numerical_param_idx = 0
        for p in self.get_neutone_parameters():
            assert p.type in self.ALLOWED_PARAM_TYPES, (
                f"Parameter type {p.type} is not allowed. "
                f"Allowed types are {self.ALLOWED_PARAM_TYPES}"
            )
            if p.type == NeutoneParameterType.CONTINUOUS:
                self.n_cont_params += 1
                self.cont_param_names.append(p.name)
                self.cont_param_indices.append(numerical_param_idx)
                numerical_param_idx += 1
            elif p.type == NeutoneParameterType.CATEGORICAL:
                self.n_cat_params += 1
                self.cat_param_names.append(p.name)
                self.cat_param_indices.append(numerical_param_idx)
                self.cat_param_n_values[p.name] = p.n_values
                numerical_param_idx += 1
            elif p.type == NeutoneParameterType.TEXT:
                self.n_text_params += 1
                self.text_param_max_n_chars.append(p.max_n_chars)
                self.text_param_default_values.append(p.default_value)

        self.n_numerical_params = self.n_cont_params + self.n_cat_params

        assert self.get_default_param_values().size(0) == self.n_numerical_params, (
            f"Default parameter values tensor first dimension must have the same  "
            f"size as the number of numerical parameters. Expected size "
            f"{self.n_numerical_params}, got {self.get_default_param_values().size(0)}"
        )
        assert self.n_numerical_params <= constants.NEUTONE_GEN_N_NUMERICAL_PARAMS, (
            f"Too many numerical (continuous and categorical) parameters. "
            f"Max allowed is {constants.NEUTONE_GEN_N_NUMERICAL_PARAMS}"
        )
        assert self.n_text_params <= constants.NEUTONE_GEN_N_TEXT_PARAMS, (
            f"Too many text parameters. "
            f"Max allowed is {constants.NEUTONE_GEN_N_TEXT_PARAMS}"
        )
        if self.n_text_params:
            self.has_text_param = True

        # This overrides the base class definitions to remove the text param or extra
        # base param since it is handled separately in the UI.
        # TODO(cm): this if statement will be removed once we get rid of the extra
        # core methods we don't need anymore
        if self.has_text_param:
            self.neutone_parameter_names = [
                p.name
                for p in self.get_neutone_parameters()
                if p.type != NeutoneParameterType.TEXT
            ]
            self.neutone_parameter_descriptions = [
                p.description
                for p in self.get_neutone_parameters()
                if p.type != NeutoneParameterType.TEXT
            ]
            self.neutone_parameter_types = [
                p.type.value
                for p in self.get_neutone_parameters()
                if p.type != NeutoneParameterType.TEXT
            ]
            self.neutone_parameter_used = [
                p.used
                for p in self.get_neutone_parameters()
                if p.type != NeutoneParameterType.TEXT
            ]

        # TODO(cm): this statement will also be removed once core is refactored
        assert len(self.get_default_param_names()) == self.n_numerical_params

        assert all(
            1 <= n <= 2 for n in self.get_audio_in_channels()
        ), "Input audio channels must be mono or stereo"
        if self.get_audio_in_labels():
            assert not len(self.get_audio_in_labels()) == len(
                self.get_audio_in_channels()
            ), "No. of input audio labels must match no. of input audio channels"

        assert (
            self.get_audio_out_channels()
        ), "Model must output at least one audio track"
        assert all(
            1 <= n <= 2 for n in self.get_audio_out_channels()
        ), "Output audio channels must be mono or stereo"
        if self.get_audio_out_labels():
            assert len(self.get_audio_out_labels()) == len(
                self.get_audio_out_channels()
            ), "No. of output audio labels must match no. of output audio channels"

    def _get_max_n_params(self) -> int:
        """
        Sets the maximum number of parameters that the model can have.
        This should not be overwritten by SDK users.
        """
        return (
            constants.NEUTONE_GEN_N_NUMERICAL_PARAMS
            + constants.NEUTONE_GEN_N_TEXT_PARAMS
        )

    def _get_numerical_default_param_values(
        self,
    ) -> List[Tuple[str, Union[float, int]]]:
        """
        Returns a list of tuples containing the name and default value of each
        numerical (float or int) parameter.
        For NonRealtimeBase models, the default values for the text parameters are
        ignored since these are not numerical and are handled separately.
        """
        result = []
        for p in self.get_neutone_parameters():
            if p.type == NeutoneParameterType.CONTINUOUS:
                result.append((p.name, p.default_value))
            elif p.type == NeutoneParameterType.CATEGORICAL:
                # Convert to float to match the type of the continuous parameters
                result.append((p.name, float(p.default_value)))
        return result

    @abstractmethod
    def get_audio_in_channels(self) -> List[int]:
        """
        Returns a list of the number of audio channels that the model expects as input.
        If the model does not require audio input, an empty list should be returned.
        Currently only supports mono and stereo audio.

        Example value: [2]
        """
        pass

    @abstractmethod
    def get_audio_out_channels(self) -> List[int]:
        """
        Returns a list of the number of audio channels that the model outputs.
        Models must output at least one audio track.
        Currently only supports mono and stereo audio.

        Example value: [2]
        """
        pass

    @abstractmethod
    def get_native_sample_rates(self) -> List[int]:
        """
        Returns a list of sample rates that the model was developed and tested
        with. If the list is empty, all common sample rates are assumed to be
        supported.

        Example value: [44100]
        """
        pass

    @abstractmethod
    def get_native_buffer_sizes(self) -> List[int]:
        """
        Returns a list of buffer sizes that the model was developed and tested
        with. If the list is empty, all common buffer sizes are assumed to be
        supported. If the model is a one-shot model, this information is ignored.

        Example value: [512, 1024, 2048]
        """
        pass

    @abstractmethod
    def is_one_shot_model(self) -> bool:
        """
        Returns True if the model is a one-shot model, i.e. it must process the entire
        input audio and / or parameters at once. If this is False, it is assumed that
        the model can process audio and parameters in blocks.
        """
        pass

    @abstractmethod
    def do_forward_pass(
        self,
        curr_block_idx: int,
        audio_in: List[Tensor],
        cont_params: Dict[str, Tensor],
        text_params: List[str],
    ) -> List[Tensor]:
        """
        SDK users can overwrite this method to implement the logic for their models.
        The inputs to this method should be treated as read-only.

        Args:
            curr_block_idx:
                The index of the current block being processed. This is only relevant if
                the model is not a one-shot model and will always be 0 otherwise.
            audio_in:
                List of torch Tensors of shape [num_channels, num_samples].
                num_samples will be one of the sizes specified in
                `get_native_buffer_sizes()` if not a one-shot model.
                The sample rate of the audio will also be one of the ones specified in
                `get_native_sample_rates()`.
            cont_params:
                Python dictionary mapping from continuous and categorical (numerical)
                parameter names (defined by the values in `get_neutone_parameters()` to
                values. By default, we aggregate the parameters to a single value per
                parameter for the current audio being processed.
                Overwrite `aggregate_continuous_params` and
                `aggregate_categorical_params` for more fine-grained control.
            text_params:
                List of strings containing the text parameters. Will be empty if the
                model does not have any text parameters.

        Returns:
            List of torch Tensors of shape [num_channels, num_samples] representing the
            output audio. The number of channels of the output audio tracks should match
            the values returned by `get_audio_out_channels()`. The sample rate of the
            output audio tracks should be the same as the input audio tracks which will
            be one of the values specified in `get_native_sample_rates()`.
        """
        pass

    def set_model_sample_rate_and_buffer_size(
        self, sample_rate: int, n_samples: int
    ) -> bool:
        """
        If the model supports dynamic sample rate or buffer size resizing, add the
        functionality here.

        Args:
            sample_rate: The sample rate to use.
            n_samples: The number of samples to resize the buffer to.

        Returns:
            bool: True if successful, False if not supported or unsuccessful.
                  Defaults to False.
        """
        return False

    def reset_model(self) -> bool:
        """
        If the model supports resetting (e.g. wiping internal state), add the
        functionality here.

        Returns:
            bool: True if successful, False if not supported or unsuccessful.
                  Defaults to False.
        """
        return False

    def aggregate_continuous_params(self, cont_params: Tensor) -> Tensor:
        """
        Aggregates parameters of shape (n_cont_params, buffer_size) to single values.

        By default we take the mean value along dimension 1 to provide a single value
        for each parameter for the current buffer.
        For more fine-grained control, override this method as required.
        """
        if self.use_debug_mode:
            assert cont_params.ndim == 2
        return tr.mean(cont_params, dim=1, keepdim=True)

    def aggregate_categorical_params(self, cat_params: Tensor) -> Tensor:
        """
        Aggregates parameters of shape (n_cat_params, buffer_size) to single values.

        By default we take the first value for each parameter for the current buffer.
        For more fine-grained control, override this method as required.
        """
        if self.use_debug_mode:
            assert cat_params.ndim == 2
        return cat_params[:, :1]

    def set_progress_percentage(self, progress_percentage: int) -> None:
        """
        Sets the progress percentage of the model.

        This can be used to indicate the progress of the model to the user. This is
        especially useful for long-running one-shot models. The progress percentage
        should be between 0 and 100.
        """
        if self.use_debug_mode:
            assert (
                0 <= progress_percentage <= 100
            ), "Progress percentage must be between 0 and 100"
        self.progress_percentage = progress_percentage

    def get_audio_in_labels(self) -> List[str]:
        """
        Returns the labels for the input audio channels which will be displayed in the
        plugin UI.
        Optional, if used, must be the same length as the number of input audio
        channels.
        """
        return []

    def get_audio_out_labels(self) -> List[str]:
        """
        Returns the labels for the output audio channels which will be displayed in the
        plugin UI.
        Optional, if used, must be the same length as the number of output audio
        channels.
        """
        return []

    def forward(
        self,
        curr_block_idx: int,
        audio_in: List[Tensor],
        numerical_params: Optional[Tensor] = None,
        text_params: Optional[List[str]] = None,
    ) -> List[Tensor]:
        """
        Internal forward pass for a NonRealtimeBase wrapped model.

        If `numerical_params` or `text_params` is None, they are populated with their
        default values when applicable.

        This method should not be overwritten by SDK users.
        """
        if text_params is None:
            text_params = self.text_param_default_values

        if self.use_debug_mode:
            assert len(audio_in) == len(self.get_audio_in_channels())
            for audio, n_ch in zip(audio_in, self.get_audio_in_channels()):
                validate_waveform(audio, n_ch == 1)
            assert len(text_params) == self.n_text_params
            if self.n_text_params:
                for text, max_n_chars in zip(text_params, self.text_param_max_n_chars):
                    if max_n_chars != -1:
                        assert (
                            len(text) <= max_n_chars
                        ), f"Input text must be shorter than {max_n_chars} characters"

        in_n = 1
        if audio_in:
            in_n = audio_in[0].size(1)

        if numerical_params is None and self.n_numerical_params > 0:
            # The default params come in as one value per block by default but for
            # compatibility with the plugin inputs we repeat them for the size of the
            # buffer. This allocates memory but should never happen in the VST since it
            # always passes parameters.
            numerical_params = self.get_default_param_values().repeat(1, in_n)

        if self.use_debug_mode:
            if numerical_params is not None:
                assert numerical_params.shape == (self.n_numerical_params, in_n)
            if not self.is_one_shot_model() and self.get_native_buffer_sizes():
                assert (
                    in_n in self.get_native_buffer_sizes()
                ), f"The model does not support a buffer size of {in_n}"

        remapped_numerical_params = {}

        if numerical_params is not None:
            # Aggregate and remap the continuous parameters
            if self.n_cont_params > 0:
                cont_params = numerical_params[self.cont_param_indices, :]
                cont_params = self.aggregate_continuous_params(cont_params)
                if self.use_debug_mode:
                    assert cont_params.ndim == 2
                    assert cont_params.size(0) == self.n_cont_params
                for idx in range(self.n_cont_params):
                    if self.use_debug_mode:
                        assert (cont_params[idx] >= 0.0).all()
                        assert (cont_params[idx] <= 1.0).all()
                    remapped_numerical_params[self.cont_param_names[idx]] = cont_params[
                        idx
                    ]
            # Aggregate and remap the categorical parameters
            if self.n_cat_params > 0:
                cat_params = numerical_params[self.cat_param_indices, :]
                cat_params = self.aggregate_categorical_params(cat_params)
                if self.use_debug_mode:
                    assert cat_params.ndim == 2
                    assert cat_params.size(0) == self.n_cat_params
                for idx in range(self.n_cat_params):
                    if self.use_debug_mode:
                        n_values = self.cat_param_n_values[self.cat_param_names[idx]]
                        assert (cat_params[idx].int() >= 0).all()
                        assert (cat_params[idx].int() <= n_values).all()
                    remapped_numerical_params[self.cat_param_names[idx]] = cat_params[
                        idx
                    ].int()

        if self.should_cancel_forward_pass():
            return []

        audio_out = self.do_forward_pass(
            curr_block_idx, audio_in, remapped_numerical_params, text_params
        )

        if self.use_debug_mode:
            assert len(audio_out) == len(self.get_audio_out_channels())
            for audio, n_ch in zip(audio_out, self.get_audio_out_channels()):
                validate_waveform(audio, n_ch == 1)

        if self.should_cancel_forward_pass():
            return []

        return audio_out

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        """
        If the model introduces an amount of delay to the output audio,
        for example due to a lookahead buffer or cross-fading, return it here
        so that it can be forwarded to the DAW to compensate. Defaults to 0.
        """
        return 0

    @tr.jit.export
    def set_sample_rate_and_buffer_size(self, sample_rate: int, n_samples: int) -> bool:
        """
        Sets the sample_rate and buffer size of the wrapper.
        This should not be overwritten by SDK users, instead please override
        the 'set_model_sample_rate_and_buffer_size' method.

        Args:
            sample_rate: The sample rate to use.
            n_samples: The number of samples to use.

        Returns:
            bool: True if 'set_model_sample_rate_and_buffer_size' is implemented and
            successful, otherwise False.
        """
        if self.use_debug_mode:
            if self.get_native_buffer_sizes():
                assert (
                    n_samples in self.get_native_buffer_sizes()
                ), f"The model does not support a native buffer size of {n_samples}"

        return self.set_model_sample_rate_and_buffer_size(sample_rate, n_samples)

    @tr.jit.export
    def reset(self) -> bool:
        """
        Resets the wrapper.
        This should not be overwritten by SDK users, instead please override the
        'reset_model' method.

        Returns:
            bool: True if 'reset_model' is implemented and successful, otherwise False.
        """
        self.set_progress_percentage(0)
        self.cancel_forward_pass_requested = False
        return self.reset_model()

    @tr.jit.export
    def get_progress_percentage(self) -> int:
        """
        Returns the progress percentage of the model.
        """
        return self.progress_percentage

    @tr.jit.export
    def should_cancel_forward_pass(self) -> bool:
        """
        Returns True if the forward pass should be cancelled.
        """
        return self.cancel_forward_pass_requested

    @tr.jit.export
    def request_cancel_forward_pass(self) -> None:
        """
        Requests to cancel the forward pass.
        """
        self.cancel_forward_pass_requested = True

    @tr.jit.export
    def is_text_model(self) -> bool:
        """
        Returns True if the model has a text parameter.
        """
        return self.has_text_param

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        # This avoids using inheritance which torchscript does not support
        preserved_attrs = self.get_core_preserved_attributes()
        preserved_attrs.extend(
            [
                "audio_in_channels",
                "audio_out_channels",
                "get_native_sample_rates",
                "get_native_buffer_sizes",
                "is_one_shot_model",
                "calc_model_delay_samples",
                "set_sample_rate_and_buffer_size",
                "reset",
                "get_progress_percentage",
                "should_cancel_forward_pass",
                "request_cancel_forward_pass",
                "is_text_model",
                "get_preserved_attributes",
                "to_metadata",
            ]
        )
        return preserved_attrs

    @tr.jit.export
    def to_metadata(self) -> NonRealtimeMetadata:
        # This avoids using inheritance which torchscript does not support
        core_metadata = self.to_core_metadata()
        return NonRealtimeMetadata(
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
            date_created=core_metadata.date_created,
            citation=core_metadata.citation,
            is_experimental=core_metadata.is_experimental,
            audio_in_channels=self.get_audio_in_channels(),
            audio_out_channels=self.get_audio_out_channels(),
            native_buffer_sizes=self.get_native_buffer_sizes(),
            native_sample_rates=self.get_native_sample_rates(),
            is_one_shot_model=self.is_one_shot_model(),
            audio_in_labels=self.get_audio_in_labels(),
            audio_out_labels=self.get_audio_out_labels(),
            is_text_model=self.is_text_model(),
        )
