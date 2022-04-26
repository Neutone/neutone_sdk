# Neutone SDK

We open source this SDK so researchers can wrap their own audio models and run them in a DAW using our [Neutone Plugin](https://neutone.space/). We offer both functionality for loading the models locally in the plugin as well as contributing them to the default list of models that is available to anyone running the plugin. We hope this will both give an opportunity for researchers to easily try their models in a DAW, but also provide creators with a collection of interesting models.

## Public Beta

The Neutone SDK is currently in public beta. The following are known shortcomings of the SDK and plugin:
- The input and output of the models in the SDK is mono, not stereo.
- We recommend setting the DAW settings to 44100 or 48000 sampling rate, 2048 buffer size
- Freezing models on save can cause instabilities
- Presets and displaying metadata information does not currently work with local model loading in the plugin

However, we do not expect big changes in the interface at this point so any models converted with the current version should be directly compatible with the release or require minimal changes.

Logs are currently dumped to `/Users/<username>/Library/Application Support/Qosmo/Neutone/neutone.log`

## Table of Contents
- [Downloading the Neutone Plugin](#download)
- [Installing the SDK](#install)
- [SDK Description](#description)
- [SDK Usage](#usage)
- [Examples](#examples)
- [Contributing to the SDK](#contributing)
- [Credits](#credits)

--- 


<a name="download"/>

## Downloading the Plugin

The Neutone Plugin is available at [https://neutone.space](https://neutone.space). We currently offer VST3 and AU plugins that can be used to load the models created with this SDK. Please visit the website for more information.


## Installing the SDK

<a name="install"/>

You can install `neutone_sdk` using pip: 

```
pip install -e "git+https://github.com/QosmoInc/neutone_sdk.git#egg=neutone_sdk"
```

<a name="description"/>

## SDK Description

The SDK provides functionality for wrapping existing PyTorch models in a way that can make them executable within the VST plugin. At its core the plugin is sending chunks of audio samples at a certain sample rate as an input and expects the same amount of samples at the output. Thus the simplest models also follow this input-output format and an example can be seen in [example_clipper.py](https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py). However, there are many different kinds of audio models so in the SDK we provide functionality for:
- On the fly STFT transforms for models that operate on spectrograms
- On the fly resampling between the DAW sample rate and the model sample rate - COMING SOON
- FIFO queues for supporting different buffer sizes as an input to the model - COMING SOON

<a name="usage"/>

## SDK Usage

### General Usage

We provide several models in the [examples](https://github.com/QosmoInc/neutone-sdk/blob/main/examples) directory. We will go through one of the simplest models, a distortion model, to illustrate.

Assume we have the following PyTorch model. Parameters will be covered later on, we will focus on the inputs and outputs for now. Assume this model receives a Tensor of shape `(2, num_samples)` as an input where `num_samples` is a parameter that can be specified.

```python
class ClipperModel(nn.Module):
    def forward(self, x: Tensor, min: float, max: float, gain: float) -> Tensor:
        return torch.clip(x, min=min*gain, max=max*gain)
```

To run this inside the VST the simplest wrapper we can write is by subclassing the WaveformToWaveformBase baseclass.
```python
class ClipperModelWrapper(WaveformToWaveformBase):
    def is_input_mono(self) -> bool:
        return False

    def is_output_mono(self) -> bool:
        return False
        
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates
              
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor] = None) -> Tensor:
        # ... Parameter unwrap logic
        x = self.model.forward(x, min, max, gain)
        return x
 ```

The method that does most of the work is `do_forward_pass`. In this case it is just a simple passthrough, but we will use it to handle parameters later on.

TODO: It currently runs as mono-mono and the `is_input_mono`, `is_output_mono` toggles do not do anything

By default the VST runs as `stereo-stereo` but when mono is desired for the model we can use the `is_input_mono` and `is_output_mono` to switch. If `is_input_mono` is toggled an averaged `(1, num_samples)` shaped Tensor will be passed as an input instead of `(2, num_samples)`. If `is_output_mono` is toggled, `do_forward_pass` is expected to return a mono Tensor that will then be duplicated across both channels at the output of the VST.

`get_native_sample_rates` and `get_native_buffer_sizes` can be used to specify any preferred sample rates or buffer sizes. In most cases these are expected to only have one element but extra flexibility is provided for more complex models. In case multiple options are provided the wrappers try to find the best one for the current setting of the DAW. Whenever the sample rate or buffer size is different from the one of the DAW a wrapper is automatically triggered that converts to the correct sampling rate or implements a FIFO queue for the requested buffer size or both. This will incur a performance penalty and potentially add delay.

### Exporting models and loading in the plugin

We provide a `save_neutone_model` helper function that saves models to disk. By default this will convert models to TorchScript and run them through a series of checks to ensure they can be loaded by the plugin. The resulting `model.nm` file can be loaded within the plugin using the `load your own` button. Read below for how to submit models to the default collection.

### Parameters

For models that can use conditioning signals we currently provide four configurable knob parameters. Within the `ClipperModelWrapper` defined above we can include the following:
```python
class ClipperModelWrapper(WaveformToWaveformBase):
    ...
    
    def get_parameters(self) -> List[Parameter]:
        return [Parameter(name="min", description="min clip threshold", default_value=0.5),
         Parameter(name="max", description="max clip threshold", default_value=1.0),
         Parameter(name="gain", description="scale clip threshold", default_value=1.0)]
         
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min = params["min"]
        max = params["max"]
        gain = params["gain"]
        x = self.model.forward(x, min, max, gain)
        return x
```

During the forward pass the `params` variable will be a dictionary like the following:
```python
{
    "min": torch.Tensor([0.5] * buffer_size),
    "max": torch.Tensor([1.0] * buffer_size),
    "gain": torch.Tensor([1.0] * buffer_size)
}
```
The keys of the dictionary are specified in the `get_parameters` function.

The parameters will always take values between 0 and 1 and the `do_forward_pass` function can be used to do any necessary rescaling before running the internal forward method of the model.

Moreover, the parameters sent by the plugin come in at a sample level granularity. By default, we take the mean of each buffer and return a single float (as a Tensor), but the `aggregate_param` method can be used to override the aggregation method. See the full clipper export file for an example of preserving this granularity.


### Submitting models

The plugin contains a default list of models aimed at creators that want to make use of them during their creative process. We encourage users to submit their models once they are happy with the results they get so they can be used by the community at large. For submission we require some additional metadata that will be used to display information about the model aimed at both creators and other researchers. This will be displayed on both the [Neutone website](https://neutone.space) and inside the plugin.

Skipping the previous clipper model, here is a more realistic example based on a random TCN overdrive model inspired by [micro-tcn](https://github.com/csteinmetz1/micro-tcn).

```python
class OverdriveModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "conv1d-overdrive.random"

    def get_model_authors(self) -> List[str]:
        return ["Nao Tokui"]

    def get_model_short_description(self) -> str:
        return "Neural distortion/overdrive effect"

    def get_model_long_description(self) -> str:
        return "Neural distortion/overdrive effect through randomly initialized Convolutional Neural Network"

    def get_technical_description(self) -> str:
        return "Random distortion/overdrive effect through randomly initialized Temporal-1D-convolution layers. You'll get different types of distortion by re-initializing the weight or changing the activation function. Based on the idea proposed by Steinmetz et al."

    def get_tags(self) -> List[str]:
        return ["distortion", "overdrive"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "https://arxiv.org/abs/2010.04237",
            "Code": "https://github.com/csteinmetz1/ronn"
        }

    def get_citation(self) -> str:
        return "Steinmetz, C. J., & Reiss, J. D. (2020). Randomized overdrive neural networks. arXiv preprint arXiv:2010.04237."
```

Check out the documentation of the methods inside [core.py](neutone_sdk/core.py), as well as the random overdrive model on the [website](https://neutone.space/models/) and in the plugin to understand where each field will be displayed.

To submit a model, please open an issue on the GitHub repository. We currently need the following:
- A short description of what the model does and how it can contribute to the community
- A link to the `model.nm` file outputted by the `save_neutone_model` helper function

<a name="examples"/>

## Examples

- Full clipper distortion model example can be found [here](examples/example_clipper.py)
- Example of a random overdrive model based on [micro-tcn](https://github.com/csteinmetz1/micro-tcn) can be found [here](examples/example_overdrive-random.py) WIP

<a name="contributing"/>

## Contributing to the SDK

We welcome any contributions to the SDK. Please add types wherever possible and use the `black` formatter for readability.

The current roadmap is:
- Finish our implementation for any combination of mono / stereo input and output
- Finish our implementation of intelligent resampling and queueing for common sample rate and buffer size combinations
- Additional testing and benchmarking of models during or after exporting
- General bug fixing and stability improvements
- Adding our own experimental neural DSP models

<a name="credits"/>

## Credits

The audacitorch project was a major inspiration for the development of the SDK. [Check it out here](
https://github.com/hugofloresgarcia/audacitorch)

