import auditioner_sdk
from auditioner_sdk.utils import get_example_inputs

def test_wav2wav(wav2wavmodel):
    outputs = [wav2wavmodel(x) for x in get_example_inputs()]
    
def test_wav2label(wav2labelmodel):
    outputs = [wav2labelmodel(x) for x in get_example_inputs()]