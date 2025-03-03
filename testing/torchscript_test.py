from typing import Dict, Any

import torch as tr
from torch import Tensor as T, nn


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = "hi"
        self.b = 1
        self.c = False
        self.d = 3.14

    @tr.jit.export
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "d": self.d,
        }

    def forward(self, x: T) -> T:
        return 2 * x


if __name__ == "__main__":
    audio = tr.randn(1, 1, 5)
    model = TestModel()
    model.eval()
    out = model(audio)
    print(out)
    print(model.get_metadata())
    traced_model = tr.jit.script(model)
    out2 = traced_model(audio)
    print(out2)
    print(traced_model.get_metadata())
