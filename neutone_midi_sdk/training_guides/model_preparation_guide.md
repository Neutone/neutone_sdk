# Preparing your model

This guide assumes that you have already trained a model for a symbolic task, on a tokenization method that
is supported by the SDK (or intend to add your own custom tokenization script).

Serializing a model in to [Torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
allows the operations to be replicated in a C++ environment,
such as those used in Digital Audio Workstations. There are two main methods to accomplish this:
- [Scripting](https://pytorch.org/docs/stable/generated/torch.jit.script.html)
- [Tracing](https://pytorch.org/docs/stable/generated/torch.jit.script.html)

Scripting is the preferable option when possible, as it is more robust to various architectures.
However, in some circumstances tracing is the only option. So far we have found that 
[HuggingFace](https://huggingface.co/docs/transformers/v4.17.0/en/serialization) models only 
support Tracing. 

### Scripting a model
In case the entire functionality of your model is encoded in the forward() function:
```angular2html
trained_model = MyModel(init_args) # trained torch.nn.Module 
scripted_model = torch.jit.script(trained_model)
torch.jit.save(scripted_model, "filename.pt")
```

You can combine multiple models / functionalities by combining them into a single forward
function of a new meta-model, and then scripting it. This is particularly useful when your 
model has a sampling process that is separate of the forward() function. 

```angular2html
class Sampler(torch.nn.Module):
    def __init__(self, args):
        super(args, self).__init__()

    def forward(self, x):
        # Here you can specify any operations needed for sampling from the output of the model
        y = x + 1
        return y

class FullModel(torch.nn.Module):
    def __init__(self, trained_model, sampler):
        super(self, trained_model, sampler).__init__()
        self.model = trained_model
        self.sampler = sampler

    def forward(self, x):
        # The full process occurs here
        logits = self.model(x)
        output = self.sampler(logits)
        return output

# Create the model
trained_model = MyModel(init_args)
sampler = Sampler(init_args)
full_model = FullModel(trained_model, sampler)

# Now you can script it all together
scripted_model = torch.jit.script(full_model)
torch.jit.save(scripted_model, "filename.pt")

```

### Tracing a model

Below is an example of how to trace a HuggingFace GPT-2 model:

```angular2html
with open(os.path.join(train_path, "config.json")) as fp:
    config = json.load(fp)
vocab_size = config["vocab_size"]
dummy_input = torch.randint(0, vocab_size, (1, 2048))
partial_model = GPT2LMHeadModel.from_pretrained(train_path, torchscript=True)
traced_model = torch.jit.trace(partial_model, example_inputs=dummy_input)
torch.jit.save(traced_model, "traced_model.pt")
```

Notably, you can combine a Traced module with other components and then Script it. This is helpful
in the above case, as the 'Generate' function requires dynamic processes that cannot be captured with
tracing. Using the combine method detailed above, you can load this Traced module alongside a custom
Generate/Sample function, and then script them all together. 

To be clear, we suggest scripting a model whenever possible. With tracing, it will record
the exact set of operations that are performed on the dummy input. There are much 
higher likelihoods of missing important parts of the model's functionality when tracing. 
