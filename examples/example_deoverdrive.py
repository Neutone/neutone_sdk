import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import librosa as lbr
from pathlib import Path
# import matplotlib.pyplot as plt


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1  = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding='same',
                            dilation=dilation, bias=False) 
        #self.bn     = nn.BatchNorm1d(out_ch)
        self.film = FiLM(out_ch, 32)
        self.relu   = nn.PReLU(out_ch)
        self.res    = nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x, p):
        x_in = x
        
        x = self.conv1(x)
        x = self.film(x, p)
        #x = self.bn(x)
        x = self.relu(x)
        x_res = self.res(x_in)

        length = x.shape[-1]
        start = (x_res.shape[-1]-length)//2
        stop  = start + length
        y = x_res[...,start:stop]
        
        x = x + y
        return x

class FiLM(nn.Module):
    def __init__(self, num_features, cond_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1) # divide input into 2 chunks
        g = g.permute(0,2,1) # 
        b = b.permute(0,2,1) #

        x = self.bn(x)      # apply BatchNorm without affine
        x = (x * g) + b     # then apply conditional affine

        return x

class EffectModel(pl.LightningModule):
    def __init__(self, ninputs=1, noutputs=1, nblocks=4, channel_growth = 2, channel_width=0, kernel_size=3, dilation_growth=2, ncondition=2):
        super().__init__()
        
        # conditioning
        self.ncondition = ncondition
        self.condition = torch.nn.Sequential(
                torch.nn.Linear(ncondition, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 32),
                torch.nn.ReLU()
            )
        
        # main model
        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs
            out_ch = in_ch * channel_growth if channel_growth > 1 else channel_width
            dilation = dilation_growth ** n

            self.blocks.append(TCNBlock(in_ch, 
                                        out_ch, 
                                        kernel_size=kernel_size, dilation=dilation))
        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)

        # for windowing
        self.prev_x = torch.empty(0, dtype=torch.float) # store previous window

    def _forward(self, x, p):    # does actually forward process
        for _, block in enumerate(self.blocks):
            x = block(x, p)
        y = torch.tanh(self.output(x)) # clipping
        return y

    def forward(self, x, c):
        stride_size = x.shape[-1] // 2

        # conditioning
        p = self.condition(c)

        # forward with the previous input
        if self.prev_x is not None and self.prev_x.shape == x.shape:
            xp = torch.cat((self.prev_x[:, :, -stride_size:], x), -1)
            yp = self._forward(xp, p)
            y = yp[:, :, stride_size:]
        else:
            # forward with only current input
            y = self._forward(x, p)
        
        # store the input as cache
        self.prev_x = x
        return y


configs = {
    'kernel_size': 13,
    'dilation_growth': 4,
    'nblocks': 9,
    'channel_growth': 0,
    'channel_width': 64,
    'optimizer': 'adam',
    'ncondition': 2,
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


from auditioner_sdk.utils import save_model, validate_metadata, \
                              get_example_inputs, test_run
from auditioner_sdk import WaveformToWaveformBase

class DeoverdriveModelWrapper(WaveformToWaveformBase):

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.condition = torch.tensor([0, 0], dtype=torch.float).reshape((1,1,-1))
    
    def do_forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        
        # do any preprocessing here! 
        # expect x to be a waveform tensor with shape (n_channels, n_samples)

        x = x.mean(0).reshape(1, 1, -1).type_as(x)
        output = self.model(x, self.condition)
        output = output.mean(0).reshape(1, -1).type_as(output)

        # do any postprocessing here!
        # the return value should be a multichannel waveform tensor with shape (n_channels, n_samples)
    
        return output

metadata = {
    'sample_rate': 44100, 
    'domain_tags': ['music', 'speech', 'environmental'],
    'short_description': 'deoverdrive model',
    'long_description':  'This description can be a max of 280 characters aaaaaaaaaaaaaaaaaaaa.',
    'tags': ['Deoverdrive'],
    'labels': ['Deoverdrive'],
    'effect_type': 'waveform-to-waveform',
    'multichannel': False,
}


if __name__ == "__main__":

    torch.set_grad_enabled(False)

    # create a root dir for our model
    root = Path('exports/deoverdrive-model')
    root.mkdir(exist_ok=True, parents=True)

    # get our model

    configs = dotdict(configs)
    model = EffectModel.load_from_checkpoint("models/epoch=52-val_loss=0.1270.ckpt", 
                ninputs=1, noutputs=1, nblocks=configs.nblocks, 
                            channel_growth=configs.channel_growth, 
                            channel_width=configs.channel_width, 
                            kernel_size=configs.kernel_size, dilation_growth=configs.dilation_growth, ncondition=configs.ncondition,
                strict=False, strategy='dp')

    # model.to(torch.float)
    model.eval()
    model.freeze()
    # print(model)

    # wrap it
    wrapper = DeoverdriveModelWrapper(model)

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