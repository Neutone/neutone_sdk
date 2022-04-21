import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from examples.example_clipper import ClipperModel, ClipperModelWrapper

torch.set_grad_enabled(False)

model = torch.jit.load("../exports/clipper.pt")

sampleRate = 48000
frequency = 1
length = 1

t = np.linspace(0, length, sampleRate * length)  #  Produces a 5 second Audio-File
tone = np.sin(frequency * 2 * np.pi * t)  #  Has frequency of 440Hz

p1 = np.linspace(0, 1, sampleRate * length)
p2 = np.linspace(0, 1, sampleRate * length)
p3 = np.linspace(0, 1, sampleRate * length)
p4 = np.linspace(0, 1, sampleRate * length)

input = torch.tensor([tone])
params = torch.tensor([p1, p2, p3, p4])

model = ClipperModel()
wrapper = ClipperModelWrapper(model)

output = wrapper.forward(input, params)

plt.plot(output.cpu().numpy()[1])
plt.show()
