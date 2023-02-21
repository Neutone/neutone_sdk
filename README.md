# Neutone SDK

We open source this SDK so researchers can wrap their own audio models and run them in a DAW using our [Neutone Plugin](https://neutone.space/). We offer both functionality for loading the models locally in the plugin as well as contributing them to the default list of models that is available to anyone running the plugin. We hope this will both give an opportunity for researchers to easily try their models in a DAW, but also provide creators with a collection of interesting models.

<a name="examples"/>

## Examples and Notebooks

- Full clipper distortion model example can be found [here](examples/example_clipper.py).
- Example of a random overdrive model based on [micro-tcn](https://github.com/csteinmetz1/micro-tcn) can be found [here](examples/example_overdrive-random.py)
- Notebooks for different models showing the entire workflow from training to exporting it using Neutone
    - [DDSP](https://colab.research.google.com/drive/15FuafmtGWEyvTOOQbN1AMIQRhGLy23Pg)
    - [RAVE](https://colab.research.google.com/drive/1hty5Bd7rJJ70hlI-5720sEY3kylNxBIt)

## v1 Release

The Neutone SDK is currently on version 1.0.0. Models exported with this version of the SDK will be incompatible with beta versions of the plugin to please make sure you are using the right version. 


The restriction for a sampling rate of 48kHz and a buffer size of 2048 is now gone and the SDK contains a wrapper that supports on the fly resampling and queueing to accomodate the requirements of both the models and the DAW thanks to great work by [@christhetree](https://github.com/christhetree).


The following are known shortcomings:
- Freezing models on save can cause instabilities, we recommend trying to save models both with and without freeze.
- Displaying metadata information does not currently work with local model loading in the plugin.
- Lookahead and on the fly STFT transforms will be implemented at the SDK level in the near future but is currently possible with additional code.
- Windows and M1 acceleration are currently not supported.

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
pip install neutone_sdk
```

<a name="description"/>

## SDK Description

The SDK provides functionality for wrapping existing PyTorch models in a way that can make them executable within the VST plugin. At its core the plugin is sending chunks of audio samples at a certain sample rate as an input and expects the same amount of samples at the output. Thus the simplest models also follow this input-output format and an example can be seen in [example_clipper.py](https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py).

<a name="usage"/>

## SDK Usage

### General Usage

We provide several models in the [examples](https://github.com/QosmoInc/neutone-sdk/blob/main/examples) directory. We will go through one of the simplest models, a distortion model, to illustrate.

Assume we have the following PyTorch model. Parameters will be covered later on, we will focus on the inputs and outputs for now. Assume this model receives a Tensor of shape `(2, buffer_size)` as an input where `buffer_size` is a parameter that can be specified.

```python
class ClipperModel(nn.Module):
    def forward(self, x: Tensor, min_val: float, max_val: float, gain: float) -> Tensor:
        return torch.clip(x, min=min_val * gain, max=max_val * gain)
```

To run this inside the VST the simplest wrapper we can write is by subclassing the WaveformToWaveformBase baseclass.
```python
class ClipperModelWrapper(WaveformToWaveformBase):
    @torch.jit.export  
    def is_input_mono(self) -> bool:
        return False
    
    @torch.jit.export
    def is_output_mono(self) -> bool:
        return False
    
    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return []  # Supports all sample rates
    
    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        # ... Parameter unwrap logic
        x = self.model.forward(x, min_val, max_val, gain)
        return x
 ```

The method that does most of the work is `do_forward_pass`. In this case it is just a simple passthrough, but we will use it to handle parameters later on.

By default the VST runs as `stereo-stereo` but when mono is desired for the model we can use the `is_input_mono` and `is_output_mono` to inform the SDK and have the inputs and outputs converted automatically. If `is_input_mono` is toggled an averaged `(1, buffer_size)` shaped Tensor will be passed as an input instead of `(2, buffer_size)`. If `is_output_mono` is toggled, `do_forward_pass` is expected to return a mono Tensor (shape `(1, buffer_size)`) that will then be duplicated across both channels at the output of the VST. This is done within the SDK to avoid unnecessary memory allocations on each pass.

`get_native_sample_rates` and `get_native_buffer_sizes` can be used to specify any preferred sample rates or buffer sizes. In most cases these are expected to only have one element but extra flexibility is provided for more complex models. In case multiple options are provided the SDK will try to find the best one for the current setting of the DAW. Whenever the sample rate or buffer size is different from the one of the DAW a wrapper is automatically triggered that converts to the correct sampling rate or implements a FIFO queue for the requested buffer size or both. This will incur a small performance penalty and add some amount of delay. In case a model is compatible with any sample rate and/or buffer_size these lists can be left empty.

This means that the tensor `x` in the `do_forward_pass` method is guaranteed to be of shape `(1 if is_input_mono else 2, buffer_size)`  where `buffer_size` will be chosen at runtime from the list provided in the `get_native_buffer_sizes` method.

### Exporting models and loading in the plugin

We provide a `save_neutone_model` helper function that saves models to disk. By default this will convert models to TorchScript and run them through a series of checks to ensure they can be loaded by the plugin. The resulting `model.nm` file can be loaded within the plugin using the `load your own` button. Read below for how to submit models to the default collection.

### Parameters

For models that can use conditioning signals we currently provide four configurable knob parameters. Within the `ClipperModelWrapper` defined above we can include the following:
```python
class ClipperModelWrapper(WaveformToWaveformBase):
    ...
    
    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [NeutoneParameter(name="min", description="min clip threshold", default_value=0.5),
                NeutoneParameter(name="max", description="max clip threshold", default_value=1.0),
                NeutoneParameter(name="gain", description="scale clip threshold", default_value=1.0)]
         
    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain = params["min"], params["max"], params["gain"]
        x = self.model.forward(x, min_val, max_val, gain)
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

To submit a model, please [open an issue on the GitHub repository](https://github.com/QosmoInc/neutone_sdk/issues/new?assignees=bogdanteleaga%2C+christhetree&labels=enhancement&template=request-add-model.md&title=%5BMODEL%5D+%3CNAME%3E). We currently need the following:
- A short description of what the model does and how it can contribute to the community
- A link to the `model.nm` file outputted by the `save_neutone_model` helper function

<a name="contributing"/>

## Contributing to the SDK

We welcome any contributions to the SDK. Please add types wherever possible and use the `black` formatter for readability.

The current roadmap is:
- Additional testing and benchmarking of models during or after exporting
- Implement lookahead and on the fly STFT transforms

<a name="credits"/>

## Credits

The audacitorch project was a major inspiration for the development of the SDK. [Check it out here](
https://github.com/hugofloresgarcia/audacitorch)

