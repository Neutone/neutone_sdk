import logging
from abc import abstractmethod
from typing import NamedTuple, Dict, List, Optional

import torch as tr
from torch import Tensor, nn

from neutone_sdk import NeutoneModel
from neutone_sdk.queues import CircularInplaceTensorQueue
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
    audio_input_channels: List[int]
    audio_output_channels: List[int]
    is_realtime: bool  # TODO(cm): move to core
    is_gpu_compatible: bool  # TODO(cm): move to core
    native_sample_rates: List[int]
    native_buffer_sizes: List[int]
    look_behind_samples: int
    offline_audio_output_samples: int
    sdk_version: str
    pytorch_version: str
    date_created: float


# TODO(cm): rename
class WaveformToWaveformBase(NeutoneModel):
    def __init__(self, model: nn.Module, use_debug_mode: bool = True) -> None:
        super().__init__(model, use_debug_mode)
        self.n_audio_inputs = len(self.get_audio_input_channels())
        self.n_audio_outputs = len(self.get_audio_output_channels())
        # TODO(cm): check if these needs to be registered like ParameterList or ModuleList
        self.model_in_buffers = [tr.zeros((n_ch, 1)) for n_ch in self.get_audio_input_channels()]
        self.in_queues = [CircularInplaceTensorQueue(n_ch, 1) for n_ch in self.get_audio_input_channels()]
        if use_debug_mode:
            if self.n_audio_inputs == 0:
                assert self.get_look_behind_samples() == 0, "At least 1 audio input must exist for look-behind"
            assert self.n_audio_outputs > 0
            if self.is_realtime():
                assert self.get_offline_audio_output_samples() == -1

        # TODO(cm): add look-behind functionality for multiple audio inputs
        # self.in_n_ch = 1 if self.is_input_mono() else 2
        # These initializations are all temporary for TorchScript typing, otherwise they would be None
        # These variables are only used if get_look_behind_samples() is greater than 0
        # self.curr_bs = -1
        # self.in_queue = CircularInplaceTensorQueue(self.in_n_ch, 1)
        # self.params_queue = CircularInplaceTensorQueue(self.MAX_N_PARAMS, 1)
        # self.model_in_buffer = tr.zeros((self.in_n_ch, 1))
        # self.params_buffer = tr.zeros((self.MAX_N_PARAMS, 1))
        # self.agg_params = tr.zeros((self.MAX_N_PARAMS, 1))

    @abstractmethod
    def get_audio_input_channels(self) -> List[int]:
        pass

    @abstractmethod
    # @tr.jit.export
    def get_audio_output_channels(self) -> List[int]:
        pass

    @abstractmethod
    def is_realtime(self) -> bool:
        pass

    @abstractmethod
    def is_gpu_compatible(self) -> bool:
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
    def do_forward_pass(self, audio_inputs: List[Tensor], params: Dict[str, Tensor]) -> List[Tensor]:
        """
        SDK users can overwrite this method to implement the logic for their models.
        The inputs to this method should be treated as read-only.

        Args:
            x:
                torch Tensor of shape [num_channels, num_samples]
                num_channels is 1 if `is_input_mono` is set to True, otherwise 2
                num_samples will be one of the sizes specified in `get_native_buffer_sizes`
                If a look-behind buffer is being used, see `get_look_behind_samples` for details on the shape of x.

                The sample rate of the audio will also be one of the ones specified in
                `get_native_sample_rates`.

                The best combination is chosen based on the DAW parameters at runtime. If
                unsure, provide only one value in the lists.
            params:
                Python dictionary mapping from parameter names (defined by the values in
                get_parameters) to values. By default, we aggregate the sample values over the
                entire buffer and provide the mean value.

                Override the `aggregate_params` method for more fine grained control.

        Returns:
            torch Tensor of shape [num_channels, num_samples]

            The shape of the output must match the shape of the input.
        """
        pass

    def get_look_behind_samples(self) -> int:
        """
        If the model requires a look-behind buffer, add the number of samples needed here. This means that the
        do_forward_pass() method will always received a tensor of shape (in_n_ch, look_behind_samples + buffer_size),
        but must still return a tensor of shape (out_n_ch, buffer_size) of the latest samples.

        We recommend avoiding using a look-behind buffer as much as possible since it makes your model less efficient
        and can result in wasted calculations during each forward pass. If using a purely convolutional model, try
        switching all the convolutions to cached convolutions instead.

        Returns:
            int: The number of look-behind samples expected by the model.
                 Defaults to 0 (no look-behind buffer)
        """
        return 0

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

    def aggregate_params(self, params: Tensor) -> Tensor:
        """
        Aggregates parameters of size (MAX_N_PARAMS, buffer_size) to single values.

        By default we take the mean value along dimension 1 to provide a single value for each parameter
        for the current buffer.

        For more fine grained control, override this method as required.
        """
        if self.use_debug_mode:
            assert params.ndim == 2
        tr.mean(params, dim=1, keepdim=True, out=self.agg_params)
        return self.agg_params

    def prepare_for_inference(self) -> None:
        """Prepare the wrapper and model for inference and to be converted to torchscript."""
        super().prepare_for_inference()
        for queue in self.in_queues:
            queue.use_debug_mode = False
            queue.eval()
        self.params_queue.use_debug_mode = False

    def forward(self, audio_inputs: List[Tensor], params: Optional[Tensor] = None) -> List[Tensor]:
        """
        Internal forward pass for a WaveformToWaveform model.

        If params is None, we fill in the default values.
        """
        if self.use_debug_mode:
            assert len(audio_inputs) == self.n_audio_inputs
            n_samples = -1
            for x, n_ch in zip(audio_inputs, self.get_audio_input_channels()):
                should_be_mono = n_ch == 1
                validate_waveform(x, should_be_mono)
                if n_samples != -1:
                    assert x.size(1) == n_samples, "All audio inputs must be the same number of samples"
                else:
                    n_samples = x.size(1)

        in_n = -1
        if self.n_audio_inputs > 0:
            in_n = audio_inputs[0].size(1)

        if params is None:
            # The default params come in as one value by default but for compatibility
            # with the plugin inputs we repeat them for the size of the buffer.
            # This allocates memory but should never happen in the VST since it always passes parameters
            params = self.get_default_param_values().repeat(1, in_n)

        if self.use_debug_mode:
            assert params.shape == (self.MAX_N_PARAMS, in_n)
            if self.n_audio_inputs > 0 and self.curr_bs != -1:
                assert in_n == self.curr_bs, (
                    f"Invalid model input audio length of {in_n} samples, "
                    f"must be of length {self.curr_bs} samples"
                )
            elif self.n_audio_inputs > 0 and self.get_native_buffer_sizes():
                assert (
                    in_n in self.get_native_buffer_sizes()
                ), f"The model does not support a buffer size of {in_n}"

            # TODO(cm): add look-behind functionality for multiple audio inputs
            # if self.get_look_behind_samples():
            #     # If a look behind buffer is being used, the queues and self.curr_bs must be initialized.
            #     # This will only potentially trigger in the forward function when just the wrapper is being tested in
            #     # python (because the SQW already sets the buffer size in its constructor when using the VST or SQW)
            #     assert self.curr_bs != -1, (
            #         "Model uses a look-behind buffer, but the incoming buffer size has not "
            #         "been set. Be sure to call `set_sample_rate_and_buffer_size` before using the model."
            #     )

        # TODO(cm): add look-behind functionality for multiple audio inputs
        # if self.get_look_behind_samples():
        #     self.in_queue.push(x)
        #     self.params_queue.push(params)
        #     n = self.in_queue.size
        #     self.model_in_buffer[:, 0:-n] = 0
        #     self.in_queue.fill(self.model_in_buffer[:, -n:])
        #     self.params_buffer[:, 0:-n] = self.get_default_param_values()
        #     self.params_queue.fill(self.params_buffer[:, -n:])
        #     x = self.model_in_buffer
        #     params = self.params_buffer

        params = self.aggregate_params(params)
        if self.use_debug_mode:
            assert params.ndim == 2
            assert params.size(0) == self.MAX_N_PARAMS
        for idx, neutone_param in enumerate(self.get_neutone_parameters()):
            self.remapped_params[neutone_param.name] = params[idx]

        audio_outputs = self.do_forward_pass(audio_inputs, self.remapped_params)

        if self.use_debug_mode:
            assert len(audio_outputs) == self.n_audio_outputs
            n_samples = -1
            for x, n_ch in zip(audio_outputs, self.get_audio_output_channels()):
                should_be_mono = n_ch == 1
                validate_waveform(x, should_be_mono)
                if n_samples != -1:
                    assert x.size(1) == n_samples, "All audio outputs must be the same number of samples"
                else:
                    n_samples = x.size(1)

        return audio_outputs

    @tr.jit.export
    def get_offline_audio_output_samples(self) -> int:
        return -1

    @tr.jit.export
    def is_resampling(self) -> bool:
        # w2w wrapper does not support resampling by default
        return False

    @tr.jit.export
    def calc_model_delay_samples(self) -> int:
        """
        If the model introduces an amount of delay to the output audio,
        for example due to a lookahead buffer or cross-fading, return it here
        so that it can be forwarded to the DAW to compensate. Defaults to 0.

        This value may change if set_buffer_size() is used to change the buffer
        size, hence this is not a constant attribute.
        """
        return 0

    @tr.jit.export
    def set_sample_rate_and_buffer_size(self, sample_rate: int, n_samples: int) -> bool:
        """
        Sets the sample_rate and buffer size of the wrapper.
        This should not be overwritten by SDK users, instead please check out
        the 'set_model_sample_rate_and_buffer_size' method.

        Args:
            sample_rate: The sample rate to use.
            n_samples: The number of samples to use.

        Returns:
            bool: True if 'set_model_sample_rate_and_buffer_size' is implemented and successful, otherwise False.
        """
        if self.use_debug_mode:
            if self.get_native_buffer_sizes():
                assert (
                    n_samples in self.get_native_buffer_sizes()
                ), f"The model does not support a native buffer size of {n_samples}"

        # TODO(cm): add look-behind functionality for multiple audio inputs
        # if self.get_look_behind_samples():
        #     if self.curr_bs == -1 or n_samples != self.curr_bs:
        #         self.curr_bs = n_samples
        #         queue_len = self.get_look_behind_samples() + self.curr_bs
        #         self.in_queue = CircularInplaceTensorQueue(
        #             self.in_n_ch, queue_len, use_debug_mode=self.use_debug_mode
        #         )
        #         self.params_queue = CircularInplaceTensorQueue(
        #             self.MAX_N_PARAMS, queue_len, use_debug_mode=self.use_debug_mode
        #         )
        #         self.model_in_buffer = tr.zeros((self.in_n_ch, queue_len))
        #         self.params_buffer = self.get_default_param_values().repeat(
        #             1, queue_len
        #         )

        return self.set_model_sample_rate_and_buffer_size(sample_rate, n_samples)

    @tr.jit.export
    def set_daw_sample_rate_and_buffer_size(
        self,
        daw_sr: int,
        daw_buffer_size: int,
        model_sr: Optional[int] = None,
        model_buffer_size: Optional[int] = None,
    ) -> None:
        """
        This method should only be used when testing the wrapper in the VST without the SampleQueueWrapper.
        It can be ignored by SDK users.
        """
        self.set_sample_rate_and_buffer_size(daw_sr, daw_buffer_size)

    @tr.jit.export
    def reset(self) -> bool:
        """
        Resets the wrapper.
        This should not be overwritten by SDK users, instead please check out the 'reset_model' method.

        Returns:
            bool: True if 'reset_model' is implemented and successful, otherwise False.
        """
        for queue in self.in_queues:
            queue.reset()
        self.in_queue.reset()
        self.params_queue.reset()
        for buffer in self.model_in_buffers:
            buffer.fill_(0)
        self.model_in_buffer.fill_(0)
        self.params_buffer[:, :] = self.get_default_param_values()
        return self.reset_model()

    @tr.jit.export
    def get_preserved_attributes(self) -> List[str]:
        # This avoids using inheritance which torchscript does not support
        preserved_attrs = self.get_core_preserved_attributes()
        preserved_attrs.extend(
            [
                "get_input_audio_channels",
                "get_output_audio_channels",
                "is_realtime",
                "is_gpu_compatible",
                "get_native_sample_rates",
                "get_native_buffer_sizes",
                "get_offline_audio_output_samples",
                "is_resampling",
                "calc_model_delay_samples",
                "set_sample_rate_and_buffer_size",
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
            date_created=core_metadata.date_created,
            citation=core_metadata.citation,
            is_experimental=core_metadata.is_experimental,
            audio_input_channels=self.get_audio_input_channels(),
            audio_output_channels=self.get_audio_output_channels(),
            is_realtime=self.is_realtime(),
            is_gpu_compatible=self.is_gpu_compatible(),
            native_buffer_sizes=self.get_native_buffer_sizes(),
            native_sample_rates=self.get_native_sample_rates(),
            look_behind_samples=self.get_look_behind_samples(),
            offline_audio_output_samples=self.get_offline_audio_output_samples(),
        )
