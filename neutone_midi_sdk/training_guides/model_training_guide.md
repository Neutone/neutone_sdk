# Model Training Guide

This guide assumes you have already chosen a dataset of MIDI files and implemented a
PyTorch model architecture. 

### Tokenization
It's very important that the model is trained on a tokenization method compatible with the 
latest version of the Neutone-MIDI SDK:

- MidiLike
- TSD
- REMI
- HVO

[Click here](https://miditok.readthedocs.io/en/latest/tokenizations.html) for full documentation
on these methods. 

### Setting Parameters
Each tokenizer has its own individual parameter settings, which are detailed in the link above. 
A key functionality in our SDK is that it is robust to all values of these settings; so you can fine-tune them exactly
to the needs of your model. For example, your 'pitch-range' could be very small if you are making a drum model.

**Important**: When setting up the tokenizer, there is an option for 'additional tokens' 
and 'special tokens'. While you can implement them for your training pipeline, we currently
do not translate any of them into actual MIDI data - i.e. a 'Chord' token will not actually
produce a chord. For this reason we generally recommend leaving them as default. 

### Saving your Settings
After tokenizing the dataset, it's important to save both the config and vocab files
in JSON format. This is used by the "Data_Preparation" pipeline to extract necessary information
for MIDI translation. 

Here is an example of tokenizing with MIDILike (more examples on Miditok doc page):

```angular2html
import argparse
import os
import shutil
import json
from pathlib import Path
from miditok import REMI

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_path", type=str, help="Path to MIDI Files")
    args = parser.parse_args()
    MIDI_PATH = args.midi_path

    # Parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': False,
                         'Rest': False,
                         'Tempo': False,
                         'Program': False,
                         'TimeSignature': False,
                         'rest_range': (2, 8),  # (half, 8 beats)
                         'nb_tempos': 32,  # nb of tempo bins
                         'tempo_range': (40, 250)}  # (min, max)
    special_tokens = ["PAD", "BOS", "EOS"]

    # Creates the tokenizer convert MIDIs to tokens
    print("#---- Tokenizing the data")
    tokens_path = Path('tokenized_data')

    # Check if the directory exists
    if tokens_path.exists() and tokens_path.is_dir():
        shutil.rmtree(tokens_path)
        os.makedirs(tokens_path)

    tokenizer = MIDILike(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens=special_tokens)
    midi_paths = list(Path(MIDI_PATH).glob('**/*.mid')) + list(Path(MIDI_PATH).glob('**/*.midi'))
    print(f"Training on {len(midi_paths)} MIDI files.\n")
    tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

    # Save tokenization settings
    tokenizer_params = Path('tokenizer_params')
    if not os.path.exists(tokenizer_params):
        os.makedirs(tokenizer_params)

    with open(os.path.join(tokenizer_params, "vocab.json"), "w") as fp:
        json.dump(tokenizer.vocab, fp)
    tokenizer.save_params(out_path="tokenizer_params/config.json")
```

Once your model is tokenized, it is time to train! As long as it is a PyTorch model that can be scripted or traced (detailed 
in the following guide, 'model_preparation_guide') then you are good to go. Happy training! 