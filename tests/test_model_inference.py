import torch
import librosa as lbr
import numpy as np
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

model = torch.jit.load("./exports/clipper/model.pt")

tone = lbr.tone(440, duration=2) * 2.0

input = torch.tensor([tone, tone])
params = torch.tensor([1.0, 1.0, 0.5])

output = model.forward(input, params)

plt.plot(output.cpu().numpy()[1])
plt.show()
