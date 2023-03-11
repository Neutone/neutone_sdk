import argparse
import json
import os
import pathlib
from typing import List, Dict, Optional

import torch
import torch as tr
import torchaudio
from torch import Tensor, nn

from neutone_sdk import NeutoneParameter, WaveformToWaveformBase

# Function that takes a file_name and optionally a path to the directory the file is expected to be, returns true if
# the file is found in the stated directory (or the current directory is dir_name = '') or False is dir/file isn't found
from neutone_sdk.utils import save_neutone_model


def file_check(file_name, dir_name=''):
    assert type(file_name) == str
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    full_path = os.path.join(*dir_name, file_name)
    return os.path.isfile(full_path)


# Function that saves 'data' to a json file. Constructs a file path is dir_name is provided.
def json_save(data, file_name, dir_name='', indent=0):
    dir_name = [dir_name] if ((type(dir_name) != list) and (dir_name)) else dir_name
    assert type(file_name) == str
    file_name = file_name + '.json' if not file_name.endswith('.json') else file_name
    full_path = os.path.join(*dir_name, file_name)
    with open(full_path, 'w') as fp:
        json.dump(data, fp, indent=indent)


class TFiLM(torch.nn.Module):
    def __init__(self,
                 nchannels,
                 nparams,
                 block_size):
        super(TFiLM, self).__init__()
        self.nchannels = nchannels
        self.nparams = nparams
        self.block_size = block_size
        self.num_layers = 1
        self.first_run = True
        self.hidden_state = (tr.Tensor(0), tr.Tensor(0))  # (hidden_state, cell_state)

        # used to downsample input
        self.maxpool = torch.nn.MaxPool1d(kernel_size=block_size,
                                          stride=None,
                                          padding=0,
                                          dilation=1,
                                          return_indices=False,
                                          ceil_mode=False)

        self.lstm = torch.nn.LSTM(input_size=nchannels + nparams,
                                  hidden_size=nchannels,
                                  num_layers=self.num_layers,
                                  batch_first=False,
                                  bidirectional=False)

    def forward(self, x, p: Optional[Tensor] = None):
        # x = [batch, nchannels, length]
        # p = [batch, nparams]
        x_in_shape = x.shape

        # pad input if it's not multiple of tfilm block size
        if (x_in_shape[2] % self.block_size) != 0:
            padding = torch.zeros(x_in_shape[0], x_in_shape[1], self.block_size - (x_in_shape[2] % self.block_size))
            x = torch.cat((x, padding), dim=-1)

        x_shape = x.shape
        nsteps = int(x_shape[-1] / self.block_size)

        # downsample signal [batch, nchannels, nsteps]
        x_down = self.maxpool(x)

        if self.nparams > 0 and p is not None:
            p_up = p.unsqueeze(-1)
            p_up = p_up.repeat(1, 1, nsteps)  # upsample params [batch, nparams, nsteps]
            x_down = torch.cat((x_down, p_up), dim=1)  # concat along channel dim [batch, nchannels+nparams, nsteps]

        # shape for LSTM (length, batch, channels)
        x_down = x_down.permute(2, 0, 1)

        # modulation sequence
        # if self.hidden_state is not None:  # state was reset
        if not self.first_run:  # state was reset
            # init hidden and cell states with zeros
            # h0 = torch.zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            # c0 = torch.zeros(self.num_layers, x.size(0), self.nchannels).requires_grad_()
            # x_norm, self.hidden_state = self.lstm(x_down, (h0.detach(), c0.detach()))  # detach for truncated BPTT
            x_norm, self.hidden_state = self.lstm(x_down, self.hidden_state)  # detach for truncated BPTT
        else:
            x_norm, self.hidden_state = self.lstm(x_down, None)
            self.first_run = False

        # put shape back (batch, channels, length)
        x_norm = x_norm.permute(1, 2, 0)

        # reshape input and modulation sequence into blocks
        x_in = torch.reshape(
            x, shape=(-1, self.nchannels, nsteps, self.block_size))
        x_norm = torch.reshape(
            x_norm, shape=(-1, self.nchannels, nsteps, 1))

        # multiply
        x_out = x_norm * x_in

        # return to original (padded) shape
        x_out = torch.reshape(x_out, shape=(x_shape))

        # crop to original (input) shape
        x_out = x_out[..., :x_in_shape[2]]

        return x_out

    # def detach_state(self):
    #     if self.hidden_state.__class__ == tuple:
    #         self.hidden_state = tuple([h.clone().detach() for h in self.hidden_state])
    #     else:
    #         self.hidden_state = self.hidden_state.clone().detach()

    def reset_state(self):
        # print("Reset Hidden State")
        self.hidden_state = None


""" 
Gated convolutional layer, zero pads and then applies a causal convolution to the input
"""


def center_crop(x: Tensor, length: int) -> Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        start = (x.size(-1) - length) // 2
        stop = start + length
        x = x[..., start:stop]
    return x


def causal_crop(x: Tensor, length: int) -> Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        stop = x.size(-1) - 1
        start = stop - length
        x = x[..., start:stop]
    return x


# TODO(cm): optimize for TorchScript
class PaddingCached(nn.Module):
    """Cached padding for cached convolutions."""

    def __init__(self, n_ch: int, padding: int) -> None:
        super().__init__()
        self.n_ch = n_ch
        self.padding = padding
        self.register_buffer("pad_buf", tr.zeros((1, n_ch, padding)))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if self.padding == 0:
            return x
        bs = x.size(0)
        if bs > self.pad_buf.size(0):  # Perform resizing once if batch size is not 1
            self.pad_buf = self.pad_buf.repeat(bs, 1, 1)
        x = tr.cat([self.pad_buf, x], dim=-1)  # concat input signal to the cache
        self.pad_buf = x[:, :, -self.padding:]  # discard old cache
        return x


class Conv1dCached(nn.Module):  # Conv1d with cache
    """Cached causal convolution for streaming."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        assert padding == 0  # We include padding in the constructor to match the Conv1d constructor
        padding = (kernel_size - 1) * dilation
        self.pad = PaddingCached(in_channels, padding)
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              (kernel_size,),
                              (stride,),
                              padding=0,
                              dilation=(dilation,),
                              bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # n_samples = x.size(-1)
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        # x = causal_crop(x, n_samples)
        return x


class GatedConv1d(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 kernel_size,
                 nparams,
                 tfilm_block_size):
        super(GatedConv1d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dilation = dilation
        self.kernal_size = kernel_size
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        # Layers: Conv1D -> Activations -> TFiLM -> Mix + Residual

        self.conv = Conv1dCached(in_channels=in_ch,
        # self.conv = nn.Conv1d(in_channels=in_ch,
                                 out_channels=out_ch * 2,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=0,
                                 dilation=dilation)

        self.tfilm = TFiLM(nchannels=out_ch,
                           nparams=nparams,
                           block_size=tfilm_block_size)

        self.mix = nn.Conv1d(in_channels=out_ch,
                             out_channels=out_ch,
                             kernel_size=1,
                             stride=1,
                             padding=0)

    def forward(self, x, p: Optional[Tensor] = None):
        # print("GatedConv1d: ", x.shape)
        residual = x

        # dilated conv
        y = self.conv(x)

        # gated activation
        z = torch.tanh(y[:, :self.out_ch, :]) * \
            torch.sigmoid(y[:, self.out_ch:, :])

        # zero pad on the left side, so that z is the same length as x
        z = torch.cat((torch.zeros(residual.shape[0],
                                   self.out_ch,
                                   residual.shape[2] - z.shape[2]),
                       z),
                      dim=2)

        # modulation
        # mock = tr.zeros_like(z)
        # mock = mock.fill_(0.5)
        z = self.tfilm(z, p)
        # z = self.tfilm(mock, p)

        x = self.mix(z)
        x = x + residual
        # x = self.mix(z) + residual

        return x, z


""" 
Gated convolutional neural net block, applies successive gated convolutional layers to the input, a total of 'layers'
layers are applied, with the filter size 'kernel_size' and the dilation increasing by a factor of 'dilation_growth' for
each successive layer.
"""


class GCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 nlayers,
                 kernel_size,
                 dilation_growth,
                 nparams,
                 tfilm_block_size):
        super(GCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.nparams = nparams
        self.tfilm_block_size = tfilm_block_size

        dilations = [dilation_growth ** l for l in range(nlayers)]

        self.layers = torch.nn.ModuleList()

        for d in dilations:
            self.layers.append(GatedConv1d(in_ch=in_ch,
                                           out_ch=out_ch,
                                           dilation=d,
                                           kernel_size=kernel_size,
                                           nparams=nparams,
                                           tfilm_block_size=tfilm_block_size))
            in_ch = out_ch

    def forward(self, x, p: Optional[Tensor] = None):
        # print("GCNBlock: ", x.shape)
        # [batch, channels, length]
        z = torch.empty([x.shape[0],
                         self.nlayers * self.out_ch,
                         x.shape[2]])

        for n, layer in enumerate(self.layers):
            x, zn = layer(x, p)
            z[:, n * self.out_ch: (n + 1) * self.out_ch, :] = zn

        return x, z


""" 
Gated Convolutional Neural Net class, based on the 'WaveNet' architecture, takes a single channel of audio as input and
produces a single channel of audio of equal length as output. one-sided zero-padding is used to ensure the network is 
causal and doesn't reduce the length of the audio.

Made up of 'blocks', each one applying a series of dilated convolutions, with the dilation of each successive layer 
increasing by a factor of 'dilation_growth'. 'layers' determines how many convolutional layers are in each block,
'kernel_size' is the size of the filters. Channels is the number of convolutional channels.

The output of the model is creating by the linear mixer, which sums weighted outputs from each of the layers in the 
model
"""


class GCNTF(torch.nn.Module):
    def __init__(self,
                 nparams=0,
                 nblocks=2,
                 nlayers=9,
                 nchannels=8,
                 kernel_size=3,
                 dilation_growth=2,
                 tfilm_block_size=128,
                 device="cpu",
                 **kwargs):
        super(GCNTF, self).__init__()
        self.nparams = nparams
        self.nblocks = nblocks
        self.nlayers = nlayers
        self.nchannels = nchannels
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.tfilm_block_size = tfilm_block_size
        self.device = device

        self.blocks = torch.nn.ModuleList()
        for b in range(nblocks):
            self.blocks.append(GCNBlock(in_ch=1 if b == 0 else nchannels,
                                        out_ch=nchannels,
                                        nlayers=nlayers,
                                        kernel_size=kernel_size,
                                        dilation_growth=dilation_growth,
                                        nparams=nparams,
                                        tfilm_block_size=tfilm_block_size))

        # output mixing layer
        self.blocks.append(
            torch.nn.Conv1d(in_channels=nchannels * nlayers * nblocks,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0))

    def forward(self, x, p: Optional[Tensor] = None):
        # print("GCN: ", x.shape)
        # x.shape = [length, batch, channels]
        # x = x.permute(1, 2, 0)  # change to [batch, channels, length]
        z = torch.empty([x.shape[0], self.blocks[-1].in_channels, x.shape[2]])

        block = self.blocks[0]
        for n, b in enumerate(self.blocks[:-1]):
            block = b
            x, zn = block(x, p)
            z[:,
                n * self.nchannels * self.nlayers:
                (n + 1) * self.nchannels * self.nlayers,
            :] = zn

        # back to [length, batch, channels]
        # return self.blocks[-1](z).permute(2, 0, 1)
        return self.blocks[-1](z)

    # def detach_states(self):
    #     # print("DETACH STATES")
    #     for layer in self.modules():
    #         if isinstance(layer, TFiLM):
    #             layer.detach_state()

    # reset state for all TFiLM layers
    def reset_states(self):
        # print("RESET STATES")
        for layer in self.modules():
            if isinstance(layer, TFiLM):
                layer.reset_state()

    # train_epoch runs one epoch of training
    def train_epoch(self,
                    dataloader,
                    loss_fcn,
                    optimiser):
        # print("TRAIN EPOCH")
        ep_losses = None

        for batch_idx, batch in enumerate(dataloader):
            # print("TRAIN BATCH")
            # reset states before starting new batch
            self.reset_states()

            # zero all gradients
            self.zero_grad()

            input, target, params = batch
            input = input.to(self.device)
            target = target.to(self.device)
            params = params.to(self.device)

            # process batch
            pred = self(input, params)
            pred = pred.to(self.device)

            # loss and backprop
            batch_losses = loss_fcn(pred, target)

            tot_batch_loss = 0
            for loss in batch_losses:
                tot_batch_loss += batch_losses[loss]

            tot_batch_loss.backward()
            optimiser.step()

            # add batch losses to epoch losses
            for loss in batch_losses:
                if ep_losses == None:
                    ep_losses = batch_losses
                else:
                    ep_losses[loss] += batch_losses[loss]

        # mean epoch losses
        for loss in ep_losses:
            ep_losses[loss] /= (batch_idx + 1)

        return ep_losses

    def val_epoch(self,
                  dataloader,
                  loss_fcn):
        val_losses = None

        # evaluation mode
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # reset states before starting new batch
                self.reset_states()

                input, target, params = batch
                input = input.to(self.device)
                target = target.to(self.device)
                params = params.to(self.device)

                # process batch
                pred = self(input, params)
                pred = pred.to(self.device)

                # loss
                batch_losses = loss_fcn(pred, target)

                tot_batch_loss = 0
                for loss in batch_losses:
                    tot_batch_loss += batch_losses[loss]

                # add batch losses to epoch losses
                for loss in batch_losses:
                    if val_losses == None:
                        val_losses = batch_losses
                    else:
                        val_losses[loss] += batch_losses[loss]

        # mean val losses
        for loss in val_losses:
            val_losses[loss] /= (batch_idx + 1)

        # back to training mode
        self.train()
        return val_losses

    def test_epoch(self,
                   dataloader,
                   loss_fcn):
        test_losses = None

        # evaluation mode
        self.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # reset states before starting new batch
                self.reset_states()

                input, target, params = batch
                input = input.to(self.device)
                target = target.to(self.device)
                params = params.to(self.device)

                # process batch
                pred = self(input, params)
                pred = pred.to(self.device)

                # loss
                batch_losses = loss_fcn(pred, target)

                tot_batch_loss = 0
                for loss in batch_losses:
                    tot_batch_loss += batch_losses[loss]

                # add batch losses to epoch losses
                for loss in batch_losses:
                    if test_losses == None:
                        test_losses = batch_losses
                    else:
                        test_losses[loss] += batch_losses[loss]

        # mean val losses
        for loss in test_losses:
            test_losses[loss] /= (batch_idx + 1)

        # back to training mode
        self.train()
        return test_losses

    def process_data(self,
                     input,
                     params):

        input = input.to(self.device)
        params = params.to(self.device)

        # evaluation mode
        self.eval()
        with torch.no_grad():
            # reset states before processing
            self.reset_states()

            out = self(x=input, p=params)

        # back to training mode
        self.train()

        # reset states before other computations
        self.reset_states()

        return out

    def compute_receptive_field(self):
        """ Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks * self.nlayers):
            dilation = self.dilation_growth ** (n % self.nlayers)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument('--nparams', type=int, default=0)
        parser.add_argument('--nblocks', type=int, default=2)
        parser.add_argument('--nlayers', type=int, default=9)
        parser.add_argument('--nchannels', type=int, default=16)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--dilation_growth', type=int, default=2)
        parser.add_argument('--tfilm_block_size', type=int, default=128)

        return parser


class MarcoModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "clipper"

    def get_model_authors(self) -> List[str]:
        return ["Andrew Fyfe"]

    def get_model_short_description(self) -> str:
        return "Audio clipper."

    def get_model_long_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_description(self) -> str:
        return "Clips the input audio between -1 and 1."

    def get_technical_links(self) -> Dict[str, str]:
        return {"Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py"}

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("min", "min clip threshold", default_value=0.15),
            NeutoneParameter("max", "max clip threshold", default_value=0.15),
            NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
            NeutoneParameter("param_4", "scale clip threshold", default_value=1.0),
        ]

    @tr.jit.export
    def is_input_mono(self) -> bool:
        return True

    @tr.jit.export
    def is_output_mono(self) -> bool:
        return True

    @tr.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]

    @tr.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        min_val, max_val, gain, param_4 = params["min"], params["max"], params["gain"], params["param_4"]
        max_val *= 10.0
        param_4 *= 10.0
        params = tr.stack([min_val, max_val, gain, param_4], dim=-1)
        x = x.unsqueeze(0)
        x = self.model.forward(x, params)
        x = x.squeeze(0)
        return x


def load_model(model_data, device):
    model_meta = model_data.pop('model_data')
    # print(model_meta)

    if model_meta["model_type"] == "gcntf":
        model = GCNTF(**model_meta, device=device)

    if 'state_dict' in model_data:
        state_dict = model.state_dict()
        for each in model_data['state_dict']:
            state_dict[each] = torch.tensor(model_data['state_dict'][each])
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    model = GCNTF(
        nparams=4,
        nblocks=1,
        nlayers=10,
        nchannels=16,
        kernel_size=3,
        dilation_growth=2,
        tfilm_block_size=128,
        device="cpu"
    )
    # print(model.compute_receptive_field())
    # audio = torch.rand((1, 1, 2048))
    # params = torch.zeros((1, 4))
    # out = model(audio, params)
    # print(out.shape)

    with open("model_best.json", "r") as in_f:
        model_data = json.load(in_f)
    # model = load_model(saved_model, "cpu")
    state_dict = model.state_dict()
    for each in model_data['state_dict']:
        new_each = each
        if "conv.weight" in each:
            new_each = each.replace("conv.weight", "conv.conv.weight")
        if "conv.bias" in each:
            new_each = each.replace("conv.bias", "conv.conv.bias")
        state_dict[new_each] = torch.tensor(model_data['state_dict'][each])
    model.load_state_dict(state_dict)

    # audio, sr = torchaudio.load("Maximum Ragga.wav")
    # audio = audio.mean(0, keepdims=True)
    # audio = audio.unsqueeze(0)
    # params = tr.zeros((1, 4))
    # wet_all = model.forward(audio, params)
    # torchaudio.save("wet_all.wav", wet_all.squeeze(0), sr)

    # bs = 2048
    # unfolded = audio.unfold(-1, bs, bs)
    # chunks = []
    # for idx in range(unfolded.size(-2)):
    #     chunk = unfolded[:, :, idx, :]
    #     proc_chunk = model.forward(chunk, params)
    #     chunks.append(proc_chunk)
    #
    # wet_chunks = tr.cat(chunks, dim=-1)
    # torchaudio.save("wet_chunks.wav", wet_chunks.squeeze(0), sr)
    # derp = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="export_model")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.output)

    wrapper = MarcoModelWrapper(model)
    save_neutone_model(
        wrapper, root_dir, freeze=False, dump_samples=True, submission=True
    )
