import torch
import librosa as lbr
import soundfile as sf
# import matplotlib.pyplot as plt

SR = 44100

torch.set_grad_enabled(False)

model = torch.jit.load("./exports/deoverdrive-model/model.pt")

input, sr = lbr.load("./assets/distorted.wav", sr=SR, mono=True)

input = torch.tensor([input])

input = input.mean(0).reshape(1, -1) # to mono and mini-batch

output = model.forward(input)

sf.write("./assets/deoverdrive.wav", output.cpu().numpy()[0], sr, subtype='PCM_24')

# plt.plot(output.cpu().numpy()[0])
# plt.show()