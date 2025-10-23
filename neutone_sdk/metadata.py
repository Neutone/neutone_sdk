import logging
import os

import requests
from jsonschema import validate, ValidationError
from jsonschema._keywords import anyOf

from neutone_sdk.audio import AudioSample

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

required_params = [
        "model_name",
        "model_authors",
        "model_version",
        "model_short_description",
        "model_long_description",
        "technical_description",
        "technical_links",
        "tags",
        "citation",
        "is_experimental",
        "sample_sound_files",
        "neutone_parameters",
        "wet_default_value",
        "dry_default_value",
        "input_gain_default_value",
        "output_gain_default_value",
        "native_sample_rates",
        "native_buffer_sizes",
        "sdk_version",
        "pytorch_version",
        "date_created",
    ]

extra_required_params_realtime = [
    "is_input_mono",
    "is_output_mono",
    "model_type",
    "look_behind_samples",
]


SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {
            "type": "string",
            "maxLength": 30,
        },
        "model_authors": {
            "type": "array",
            "maxItems": 5,
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "model_version": {"type": "string"},
        "model_short_description": {"type": "string", "maxLength": 150},
        "model_long_description": {"type": "string", "maxLength": 500},
        "technical_description": {"type": "string", "maxLength": 500},
        "technical_links": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
            },
            "maxItems": 3,
        },
        "tags": {
            "type": "array",
            "maxItems": 7,
            "items": {"type": "string"},
            "uniqueItems": True,
            "maxLength": 15,
        },
        "citation": {
            "type": "string",
            "maxLength": 300,
        },
        "is_experimental": {
            "type": "boolean",
        },
        "model_id": {"type": "string"},
        "file_size": {"type": "integer"},
        "sample_sound_files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["in", "out"],
                "properties": {
                    "in": {"type": "string"},
                    "out": {"type": "string"},
                },
            },
            "maxItems": 3,
        },
        "neutone_parameters": {
            "type": "object",
            anyOf: [
                {"required": ["p1"]},
                {"required": ["p1", "p2"]},
                {"required": ["p1", "p2", "p3"]},
                {"required": ["p1", "p2", "p3", "p4"]},
            ],
            "properties": {
                "p1": {"$ref": "#/definitions/neutoneParameter"},
                "p2": {"$ref": "#/definitions/neutoneParameter"},
                "p3": {"$ref": "#/definitions/neutoneParameter"},
                "p4": {"$ref": "#/definitions/neutoneParameter"},
            },
        },
        "wet_default_value": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "dry_default_value": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "input_gain_default_value": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "output_gain_default_value": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "is_input_mono": {
            "type": "boolean",
        },
        "is_output_mono": {
            "type": "boolean",
        },
        "model_type": {
            "type": "string",
            "enum": ["mono-mono", "mono-stereo", "stereo-mono", "stereo-stereo"],
        },
        "native_sample_rates": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 0,
                "maximum": 384000,
            },
            "uniqueItems": True,
        },
        "native_buffer_sizes": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1,
            },
            "uniqueItems": True,
        },
        "look_behind_samples": {
            "type": "integer",
            "minimum": 0,
        },
        "sdk_version": {"type": "string"},
        "pytorch_version": {"type": "string"},
        "date_created": {"type": "number"},
    },
    "definitions": {
        "neutoneParameter": {
            "type": "object",
            "required": ["name", "description", "default_value", "used", "type"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "default_value": {"type": ["integer", "number", "string"]},
                "used": {"type": "boolean"},
                "type": {"type": "string", "enum": ["continuous"]},
                "max_n_chars": {"type": ["null", "integer"], "minimum": -1},
                "n_values": {"type": ["null", "integer"], "minimum": 2},
                "labels": {"type": ["null", "array"], "items": {"type": "string"}},
            },
        }
    },
    "required": required_params,
}


def validate_metadata(metadata: dict, realtime: bool) -> bool:
    if realtime:
        SCHEMA["required"] = required_params + extra_required_params_realtime
        SCHEMA["properties"]["native_sample_rates"]["items"]["maximum"] = 96000
    try:
        validate(instance=metadata, schema=SCHEMA)
    except ValidationError as err:
        log.error(err)
        raise err

    # Check links return 200
    for link in metadata["technical_links"].values():
        try:
            code = requests.head(link, allow_redirects=True).status_code
            if code != 200:
                log.error(f"Cannot access link {link}")
        except requests.exceptions.ConnectionError:
            log.error(f"Cannot access link {link}")

    # Check we can extract mp3s from the samples
    for audio_sample_pair in metadata["sample_sound_files"]:
        AudioSample.from_b64(audio_sample_pair["in"])
        AudioSample.from_b64(audio_sample_pair["out"])

    return True
