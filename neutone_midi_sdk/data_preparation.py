from typing import Dict, List, Tuple
from neutone_midi_sdk.tokenization import TokenData


def prepare_token_data(token_type: str,
                       midi_to_token_vocab: Dict[str, int],
                       config) \
        -> TokenData:
    """
    While the forward method is tokenizing data, it requires knowledge of the 'available'
    values. For example, if velocity is constrained to 32 values (instead of 127), it needs
    a list of these available values. To save on computational time within the plugin environment,
    this function is run once during model wrapping, to extract all necessary values. They are
    returned in a Dictionary of Lists, each of which can quickly be accessed in real-time.

    Each tokenization has its own set of unique token types.
    MIDILike: [TimeShift, Pitch, Velocity]
    """

    assert token_type in ["MIDILike", "TSD", "REMI", "Custom"], "Incorrect tokenization method specified."

    if token_type == "MIDILike":
        token_data: TokenData = prepare_MIDILike_data(midi_to_token_vocab)
        return token_data

    elif token_type == "TSD":
        token_data: TokenData = prepare_TSD_data(midi_to_token_vocab)
        return token_data

    elif token_type == "REMI":
        token_data: TokenData = prepare_REMI_data(midi_to_token_vocab, config)
        return token_data

    else:
        print("incorrect tokenization type specified, recommend retrying with a different value specified")
        token_data: TokenData = prepare_MIDILike_data(midi_to_token_vocab)
        return token_data


##### Utility Functions #####
def extract_timing_tokens(tokens: List[str], identifier: str) -> Tuple[List[str], List[float]]:
    """
    Extract information from tokens related to shifts in time, i.e. TimeShift and Duration.
    They must follow the format of "Type_beat.subdivision.granularity", i.e.
    "TimeShift_1.2.4" = Timeshift of 1 beat and 2 samples of 4-per-beat (16ths).
    The tokenizer needs both the string version (for tokenizing), and
    values that convert this to float timing in MIDI format, i.e. 1.2.4 = 1.5 (1.5 quarter notes)
    Returns a tuple containing lists of both string and float formats.
    """
    timing_token_strings: List[str] = []
    for token in tokens:
        if identifier in token:
            timing_token_strings.append(token)

    reduced_tokens: List[str] = [token.split("_")[1] for token in timing_token_strings]
    timing_token_floats: List[float] = []
    for value in reduced_tokens:
        number = int(value.split(".")[0]) + int(value.split(".")[1]) * (1 / int(value.split(".")[2]))
        timing_token_floats.append(number)

    output: Tuple[List[str], List[float]] = (timing_token_strings, timing_token_floats)
    return output

def extract_timing_tokens(tokens: List[str], identifier: str) -> Tuple[List[str], List[float]]:
    """
    Extract information from tokens related to shifts in time, i.e. TimeShift and Duration.
    They must follow the format of "Type_beat.subdivision.granularity", i.e.
    "TimeShift_1.2.4" = Timeshift of 1 beat and 2 samples of 4-per-beat (16ths).
    The tokenizer needs both the string version (for tokenizing), and
    values that convert this to float timing in MIDI format, i.e. 1.2.4 = 1.5 (1.5 quarter notes)
    Returns a tuple containing lists of both string and float formats.
    """
    timing_token_strings: List[str] = []
    for token in tokens:
        if identifier in token:
            timing_token_strings.append(token)

    reduced_tokens: List[str] = [token.split("_")[1] for token in timing_token_strings]
    timing_token_floats: List[float] = []
    for value in reduced_tokens:
        number = int(value.split(".")[0]) + int(value.split(".")[1]) * (1 / int(value.split(".")[2]))
        timing_token_floats.append(number)

    output: Tuple[List[str], List[float]] = (timing_token_strings, timing_token_floats)
    return output


def extract_value_tokens(tokens: List[str], identifier: str) -> List[int]:
    """
    Designed to extract any token that has a string and an integer, i.e.
    ["Pitch_23"] or ["Velocity_99"]. As this is a common format throughout Miditok,
    the function can be called on a variety of use cases.
    """
    values: List[int] = []
    for token in tokens:
        if identifier in token:
            values.append(int(token.split("_")[1]))

    return values


## Tokenization Preparation Functions ##
def prepare_MIDILike_data(midi_to_token_vocab: Dict[str, int]) \
        -> TokenData:

    tokens = [key for key in midi_to_token_vocab.keys()]

    # Get lists of available values
    timeshift_tokens: Tuple[List[str], List[float]] = extract_timing_tokens(tokens, identifier="TimeShift")
    pitch_values: List[int] = extract_value_tokens(tokens, identifier="NoteOn")
    velocity_values: List[int] = extract_value_tokens(tokens, identifier="Velocity")

    token_strings: Dict[str, List[str]] = {"timeshift_strings": timeshift_tokens[0]}
    token_floats: Dict[str, List[float]] = {"timeshift_floats": timeshift_tokens[1]}
    token_ints: Dict[str, List[int]] = {"pitch_values": pitch_values, "velocity_values": velocity_values}

    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)

    return token_data

def prepare_TSD_data(midi_to_token_vocab: Dict[str, int]) \
    -> TokenData:

    tokens = [key for key in midi_to_token_vocab.keys()]

    # Get lists of available values
    timeshift_tokens: Tuple[List[str], List[float]] = extract_timing_tokens(tokens, identifier="TimeShift")
    duration_tokens: Tuple[List[str], List[float]] = extract_timing_tokens(tokens, identifier="Duration")
    pitch_values: List[int] = extract_value_tokens(tokens, identifier="Pitch")
    velocity_values: List[int] = extract_value_tokens(tokens, identifier="Velocity")

    token_strings: Dict[str, List[str]] = {"timeshift_strings": timeshift_tokens[0],
                                           "duration_strings": duration_tokens[0]}

    token_floats: Dict[str, List[float]] = {"timeshift_floats": timeshift_tokens[1],
                                            "duration_floats": duration_tokens[1]}

    token_ints: Dict[str, List[int]] = {"pitch_values": pitch_values,
                                        "velocity_values": velocity_values}

    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)

    return token_data


def prepare_REMI_data(midi_to_token_vocab: Dict[str, int], config) \
    -> TokenData:

    tokens = [key for key in midi_to_token_vocab.keys()]

    # Determine the granularity of the "Position" tokens, defined by the (0_N: res) beat res in Miditok config file
    for k, v in config["beat_res"].items():
        if "0" in (k):
            pos_granularity: List[float] = [float(1/v)]
            break

    position_values: List[int] = extract_value_tokens(tokens, identifier="Position")
    position_floats: List[float] = list()
    position_strings: List[str] = list()
    for token in tokens:
        if "Position" in token:
            position_strings.append(token)
            value = int(token.split("_")[1])
            position_floats.append(float(value * pos_granularity[0]))

    pitch_values: List[int] = extract_value_tokens(tokens, identifier="Pitch")
    velocity_values: List[int] = extract_value_tokens(tokens, identifier="Velocity")
    duration_tokens: Tuple[List[str], List[float]] = extract_timing_tokens(tokens, identifier="Duration")

    token_strings: Dict[str, List[str]] = {"position_strings": position_strings,
                                           "duration_strings": duration_tokens[0]}

    token_floats: Dict[str, List[float]] = {"pos_granularity": pos_granularity,
                                            "position_floats": position_floats,
                                            "duration_floats": duration_tokens[1]}

    token_ints: Dict[str, List[int]] = {"pitch_values": pitch_values,
                                        "velocity_values": velocity_values,
                                        "pos_values": position_values}

    token_data: TokenData = TokenData(token_strings, token_floats, token_ints)

    return token_data