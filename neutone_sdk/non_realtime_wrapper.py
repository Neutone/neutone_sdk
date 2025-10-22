import json
import logging
import os
from enum import Enum
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any

import torch as tr
from torch import Tensor, nn

from neutone_sdk import (
    NeutoneModel,
    constants,
    NeutoneParameterType,
    utils,
)
from neutone_sdk.utils import validate_waveform

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class TokenizerType(Enum):
    JSON = "JSON"  # huggingface tokenizer.json FromBlobJSON
    SENTENCEPIECE = "SentencePiece"  # FromBlobSentencePiece
    RWKVWORLD = "RWKVWorld"  # FromBlobRWKVWorld


class NonRealtimeBase(NeutoneModel):
    ALLOWED_PARAM_TYPES = {
        NeutoneParameterType.CONTINUOUS,
        NeutoneParameterType.CATEGORICAL,
        NeutoneParameterType.TEXT,
        NeutoneParameterType.TOKENS,
    }
    # TorchScript typing does not support instance attributes, so we need to type them
    # as class attributes. This is required for supporting models with no parameters.
    # (https://github.com/pytorch/pytorch/issues/51041#issuecomment-767061194)
    # From NeutoneModel, sometimes TorchScript complains if there are not redefined here
    neutone_parameters_metadata: Dict[
        str, Dict[str, Union[int, float, str, bool, List[str], List[int]]]
    ]
    text_param_max_n_chars: List[int]
    text_param_default_values: List[str]
    tokens_param_max_n_tokens: List[int]
    tokens_param_default_values: List[List[int]]

    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        """
        Wraps a PyTorch model for use in a non-realtime context.
        Compatible with the Neutone Gen plugin.
        """
        super().__init__(model, use_debug_mode)
        self.realtime = False
        self.default_daw_sr = constants.DEFAULT_DAW_SR
        self.default_daw_bs = constants.DEFAULT_DAW_BS
        self.current_model_sample_rate = utils.select_best_model_sr(
            self.default_daw_sr, self.get_native_sample_rates()
        )
        self.current_model_buffer_size = utils.select_best_model_buffer_size(
            self.default_daw_bs, self.get_native_buffer_sizes()
        )

        self.progress_percentage = 0.0
        self.cancel_forward_pass_requested = False
        self.has_text_param = False
        self.has_tokens_param = False

        self.n_text_params = 0
        self.text_param_max_n_chars = []
        self.text_param_default_values = []

        self.n_tokens_params = 0
        self.tokens_param_max_n_tokens = []
        self.tokens_param_default_values = []

        self.has_tokenizer = False

        self.numerical_params = nn.ModuleList()
        for p in self.get_neutone_parameters():
            assert p.type in self.ALLOWED_PARAM_TYPES, (
                f"Parameter type {p.type} is not allowed. "
                f"Allowed types are {self.ALLOWED_PARAM_TYPES}"
            )
            if p.type == NeutoneParameterType.CONTINUOUS:
                self.numerical_params.append(p)
            elif p.type == NeutoneParameterType.CATEGORICAL:
                self.numerical_params.append(p)
            elif p.type == NeutoneParameterType.TEXT:
                self.n_text_params += 1
                self.text_param_max_n_chars.append(p.max_n_chars)
                self.text_param_default_values.append(p.default_value)
            elif p.type == NeutoneParameterType.TOKENS:
                self.n_tokens_params += 1
                self.tokens_param_max_n_tokens.append(p.max_n_tokens)
                self.tokens_param_default_values.append(p.default_value)
        self.n_numerical_params = len(self.numerical_params)

        assert (
            self.get_numerical_params_default_values_0to1().size(0)
            == self.n_numerical_params
        ), (
            f"Default parameter values tensor first dimension must have the same  "
            f"size as the number of numerical parameters. Expected size "
            f"{self.n_numerical_params}, "
            f"got {self.get_numerical_params_default_values_0to1().size(0)}"
        )
        assert self.n_numerical_params <= constants.NEUTONE_GEN_N_NUMERICAL_PARAMS, (
            f"Too many numerical (continuous and categorical) parameters. "
            f"Max allowed is {constants.NEUTONE_GEN_N_NUMERICAL_PARAMS}"
        )
        assert self.n_text_params <= constants.NEUTONE_GEN_N_TEXT_PARAMS, (
            f"Too many text parameters. "
            f"Max allowed is {constants.NEUTONE_GEN_N_TEXT_PARAMS}"
        )
        assert self.n_tokens_params <= constants.NEUTONE_GEN_N_TOKENS_PARAMS, (
            f"Too many tokens parameters. "
            f"Max allowed is {constants.NEUTONE_GEN_N_TOKENS_PARAMS}"
        )
        if self.n_text_params:
            self.has_text_param = True
        if self.n_tokens_params:
            self.has_tokens_param = True

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

        self.metadata_json_str = json.dumps(
            self.to_metadata(), indent=4, sort_keys=True
        )

    def _get_max_n_params(self) -> int:
        """
        Sets the maximum number of parameters that the model can have.
        This should not be overwritten by SDK users.
        """
        return (
            constants.NEUTONE_GEN_N_NUMERICAL_PARAMS
            + constants.NEUTONE_GEN_N_TEXT_PARAMS
            + constants.NEUTONE_GEN_N_TOKENS_PARAMS
        )

    def _get_numerical_params_default_values_0to1(
        self,
    ) -> Tensor:
        """
        Returns a float tensor with the default values of the numerical parameters
        in the range [0, 1].
        For NonRealtimeBase models, the default values for the text parameters are
        ignored since these are not numerical and are handled separately.
        """
        result = []
        for p in self.get_neutone_parameters():
            if p.type == NeutoneParameterType.CONTINUOUS:
                result.append(p.default_value_0to1)
            elif p.type == NeutoneParameterType.CATEGORICAL:
                result.append(p.default_value_0to1)
        result = tr.tensor(result)
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
        numerical_params: Dict[str, Tensor],
        text_params: List[str],
        tokens_params: List[List[int]],
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
            numerical_params:
                Python dictionary mapping from continuous and categorical (numerical)
                parameter names (defined by the values in `get_neutone_parameters()` to
                values. By default, we aggregate the parameters to a single value per
                parameter for the current audio being processed.
                Overwrite `aggregate_numerical_params_0to1` for fine-grained control.
            text_params:
                List of strings containing the text parameters. Will be empty if the
                model does not have any text parameters.
            tokens_params:
                List of list of ints containing the tokens. Will be empty if the
                model does not have any tokens parameters.

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

    def aggregate_numerical_params_0to1(self, numerical_params_0to1: Tensor) -> Tensor:
        """
        Aggregates parameters of shape (n_numerical_params, buffer_size) to single
        values.

        By default we take the mean value along dimension 1 to provide a single value
        for each parameter for the current buffer.
        For more fine-grained control, override this method as required.
        """
        if self.use_debug_mode:
            assert numerical_params_0to1.ndim == 2
        return tr.mean(numerical_params_0to1, dim=1, keepdim=True)

    def set_progress_percentage(self, progress_percentage: float) -> None:
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

    def has_progress_percentage(self) -> bool:
        """
        Returns True if the model sets the progress percentage of the model during
        forward pass.
        If this is False and the model is a oneshot model, the plugin should estimate the progress based on last run.
        """
        return True

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
        numerical_params_0to1: Optional[Tensor] = None,
        text_params: Optional[List[str]] = None,
        tokens_params: Optional[List[List[int]]] = None,
    ) -> List[Tensor]:
        """
        Internal forward pass for a NonRealtimeBase wrapped model.

        If `numerical_params`, `text_params` or `tokens_params` is None, they are populated
        with their default values when applicable.
        This method should not be overwritten by SDK users.
        """
        self.set_progress_percentage(0.0)  # Reset progress percentage

        if text_params is None:
            text_params = self.text_param_default_values
        if tokens_params is None:
            tokens_params = self.tokens_param_default_values

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
            assert len(tokens_params) == self.n_tokens_params
            if self.n_tokens_params:
                for tokens, max_n_tokens in zip(
                    tokens_params, self.tokens_param_max_n_tokens
                ):
                    if max_n_tokens != -1:
                        assert (
                            len(tokens) <= max_n_tokens
                        ), f"Input tokens must be shorter than {max_n_tokens}"

        # Default value for in_n is the current model buffer size
        in_n = self.current_model_buffer_size
        if numerical_params_0to1 is not None:
            # If numerical_params_0to1 is provided, we use its size to determine in_n
            in_n = numerical_params_0to1.size(1)
        if audio_in:
            # If audio_in is provided, we use its size to determine in_n
            in_n = audio_in[0].size(1)

        if numerical_params_0to1 is None and self.n_numerical_params > 0:
            # The default params come in as one value per block by default but for
            # compatibility with the plugin inputs we repeat them for the size of the
            # buffer. This allocates memory but should never happen in the VST since it
            # always passes parameters.
            numerical_params_0to1 = (
                self.get_numerical_params_default_values_0to1().repeat(1, in_n)
            )

        if self.use_debug_mode:
            if numerical_params_0to1 is not None:
                assert numerical_params_0to1.size(0) == self.n_numerical_params
                if audio_in:
                    assert numerical_params_0to1.size(1) == in_n
            if not self.is_one_shot_model() and self.get_native_buffer_sizes():
                assert (
                    in_n in self.get_native_buffer_sizes()
                ), f"The model does not support a buffer size of {in_n}"

        remapped_numerical_params = {}
        if numerical_params_0to1 is not None:
            # Aggregate and remap the numerical parameters
            numerical_params_0to1 = self.aggregate_numerical_params_0to1(
                numerical_params_0to1
            )
            if self.use_debug_mode:
                assert numerical_params_0to1.ndim == 2
                assert numerical_params_0to1.size(0) == self.n_numerical_params
                assert (numerical_params_0to1 >= 0.0).all()
                assert (numerical_params_0to1 <= 1.0).all()
            for idx, curr_param in enumerate(self.numerical_params):
                curr_val_0to1 = numerical_params_0to1[idx]
                curr_val = curr_param.from_0to1(curr_val_0to1)
                remapped_numerical_params[curr_param.name] = curr_val

        if self.should_cancel_forward_pass():
            return []

        audio_out = self.do_forward_pass(
            curr_block_idx,
            audio_in,
            remapped_numerical_params,
            text_params,
            tokens_params,
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
            assert sample_rate > 0
            assert n_samples > 0
            if self.get_native_sample_rates():
                assert (
                    sample_rate in self.get_native_sample_rates()
                ), f"The model does not support a sample rate of {sample_rate}"
            if self.get_native_buffer_sizes():
                assert (
                    n_samples in self.get_native_buffer_sizes()
                ), f"The model does not support a native buffer size of {n_samples}"
        self.current_model_sample_rate = sample_rate
        self.current_model_buffer_size = n_samples
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
        self.current_model_sample_rate = utils.select_best_model_sr(
            self.default_daw_sr, self.get_native_sample_rates()
        )
        self.current_model_buffer_size = utils.select_best_model_buffer_size(
            self.default_daw_bs, self.get_native_buffer_sizes()
        )
        self.set_progress_percentage(0.0)
        self.cancel_forward_pass_requested = False
        return self.reset_model()

    @tr.jit.export
    def get_progress_percentage(self) -> float:
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
    def is_tokens_model(self) -> bool:
        """
        Returns True if the model has a tokens parameter.
        """
        return self.has_tokens_param

    @tr.jit.export
    def get_current_model_sample_rate(self) -> int:
        """
        Returns the current sample rate of the model if it has been set, else None.
        """
        return self.current_model_sample_rate

    @tr.jit.export
    def get_current_model_buffer_size(self) -> int:
        """
        Returns the current buffer size of the model if it has been set, else None.
        """
        return self.current_model_buffer_size

    @tr.jit.export
    def get_model_bpm(self) -> Optional[int]:
        """
        Returns the BPM the model was trained on if there is one.
        """
        return None

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
                "has_progress_percentage",
                "should_cancel_forward_pass",
                "request_cancel_forward_pass",
                "is_text_model",
                "is_tokens_model",
                "get_current_model_sample_rate",
                "get_current_model_buffer_size",
                "get_model_bpm",
                "get_preserved_attributes",
                "to_metadata",
                "get_metadata_json",
                "get_tokenizer_str",
                "get_tokenizer_type",
            ]
        )
        return preserved_attrs

    @tr.jit.export
    def to_metadata(self) -> Dict[str, Any]:
        # This avoids using inheritance which torchscript does not support
        core_metadata = self.to_core_metadata()
        core_metadata["audio_in_channels"] = self.get_audio_in_channels()
        core_metadata["audio_out_channels"] = self.get_audio_out_channels()
        core_metadata["native_buffer_sizes"] = self.get_native_buffer_sizes()
        core_metadata["native_sample_rates"] = self.get_native_sample_rates()
        core_metadata["is_one_shot_model"] = self.is_one_shot_model()
        core_metadata["audio_in_labels"] = self.get_audio_in_labels()
        core_metadata["audio_out_labels"] = self.get_audio_out_labels()
        core_metadata["is_text_model"] = self.is_text_model()
        core_metadata["is_tokens_model"] = self.is_tokens_model()
        core_metadata["model_bpm"] = self.get_model_bpm()
        return core_metadata

    @tr.jit.export
    def get_metadata_json(self) -> str:
        return self.metadata_json_str

    @tr.jit.export
    def get_tokenizer_str(self) -> str:
        return ""

    @tr.jit.export
    def get_tokenizer_type(self) -> Optional[str]:
        return None


class NonRealtimeTokenizerBase(NonRealtimeBase):

    def __init__(
        self,
        model: nn.Module,
        tokenizer_str: str,
        tokenizer_type: TokenizerType,
        use_debug_mode: bool = True,
    ) -> None:
        super().__init__(model, use_debug_mode)
        self.tokenizer_str = tokenizer_str  # BASE 64
        ALLOWED_TOKENIZER_TYPES = {
            TokenizerType.JSON,
            TokenizerType.RWKVWORLD,
            TokenizerType.SENTENCEPIECE,
        }
        assert tokenizer_type in ALLOWED_TOKENIZER_TYPES, (
            f"Parameter type {tokenizer_type} is not allowed. "
            f"Allowed types are {ALLOWED_TOKENIZER_TYPES}"
        )
        self.tokenizer_type = tokenizer_type.value
        self.has_tokenizer = True

    @tr.jit.export
    def get_tokenizer_str(self) -> str:
        return self.tokenizer_str

    @tr.jit.export
    def get_tokenizer_type(self) -> Optional[str]:
        return self.tokenizer_type
