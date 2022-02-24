import torch
import librosa as lbr
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

model = torch.jit.load("./exports/clipper-model/model.pt")

tone = lbr.tone(440, duration=2) * 2.0

input = torch.tensor([tone, tone])

output = model.forward(input)

plt.plot(output.cpu().numpy()[1])
plt.show()

