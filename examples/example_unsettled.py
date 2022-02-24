import torch
import torch.nn as nn
from auditioner_sdk import WaveformToWaveformBase


class UnsettledModelWrapper(WaveformToWaveformBase):

    def __init__(self, module: nn.Module):
        super().__init__(module)
        
        self.unsettled_tensor = torch.zeros(1, 2048)
        self.output = torch.zeros(2, 2048)
        self.alpha =  torch.ones(1)

        self.register_buffer("alpha_param", self.alpha)

    @torch.jit.export
    def alpha(self, x: torch.Tensor):
        self.alpha = x
    
    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        
        # do any preprocessing here! 
        # expect x to be a waveform tensor with shape (n_channels, n_samples)
        # self.unsettled_tensor[0] = torch.sum(x, 1)
        self.unsettled_tensor[0] = x.mean(0)
        output = self.model(self.unsettled_tensor[0], self.alpha)

        self.output[0] = output
        self.output[1] = output

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
    
        return self.output

metadata = {
    'sample_rate': 96000, 
    'domain_tags': ['music', 'speech', 'environmental'],
    'short_description': 'Unsettled model test',
    'long_description':  'This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.',
    'tags': ['Unsettled'],
    'labels': ['Unsettled'],
    'effect_type': 'waveform-to-waveform',
    'multichannel': False,
}

from pathlib import Path
from auditioner_sdk.utils import save_model, validate_metadata, \
                              get_example_inputs, test_run


if __name__ == "__main__":
    
    torch.set_grad_enabled(False)

    # create a root dir for our model
    root = Path('exports/unsettled-model')
    root.mkdir(exist_ok=True, parents=True)

    # get our model
    # model = UnsettledModel("models/unsettled.pt")
    model = torch.jit.load("models/unsettled_fast.pt")
    # wrap it
    wrapper = UnsettledModelWrapper(model)

    # serialize it using torch.jit.script, torch.jit.trace,
    # or a combination of both. 

    # option 1: torch.jit.script 
    # using torch.jit.script is preferred for most cases, 
    # but may require changing a lot of source code
    serialized_model = torch.jit.script(wrapper)

    # option 2: torch.jit.trace
    # using torch.jit.trace is typically easier, but you
    # need to be extra careful that your serialized model behaves 
    # properly after tracing
    # example_inputs = get_example_inputs()
    # serialized_model = torch.jit.trace(wrapper, example_inputs[0], 
    #                                     check_inputs=example_inputs)

    # take your model for a test run!
    test_run(serialized_model)

    # check that we created our metadata correctly
    success, msg = validate_metadata(metadata)
    assert success

    # save!
    save_model(serialized_model, metadata, root)