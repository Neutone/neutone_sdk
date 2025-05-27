import argparse
import json
import logging
import os
import argparse, json, logging, os, base64, io, tempfile
from typing import List, Dict

import torch
import torchaudio

from neutone_sdk import (
    NeutoneParameter,
    DiscreteTokensNeutoneParameter,
    ContinuousNeutoneParameter,
)
from neutone_sdk.non_realtime_sqw import NonRealtimeSampleQueueWrapper
from neutone_sdk.non_realtime_wrapper import NonRealtimeTokenizerBase, TokenizerType

"""
To run this script, you will need to install tokenizers library and also protobuf 
if you are using the sentencepiece tokenizer.
"""

TOK_TYPE = TokenizerType.SENTENCEPIECE

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

# class MusicGenWrapperNoTok(nn.Module):
#     def __init__(self, text_encoder, lm, audio_decoder, enc_to_dec_proj, logits_processor, pad_token_id: int, decoder_start_token_id: int, delay_mask_fn, num_codebooks: int, audio_channels: int):
#         super().__init__()
#         self.text_encoder = text_encoder
#         self.audio_decoder = audio_decoder
#         self.lm = lm
#         self.decoder_start_token_id = decoder_start_token_id
#         self.delay_mask_fn = delay_mask_fn
#         self.num_codebooks = num_codebooks
#         self.audio_channels = audio_channels
#         self.enc_to_dec_proj = enc_to_dec_proj
#         self.logits_processor = logits_processor
#         self.pad_token_id = pad_token_id

#     def prepare_text_encoder_kwargs_for_generation(self, input_ids):
#         encoder_attention_mask = torch.where(input_ids==0, 0, 1)
#         encoder_outputs = self.text_encoder(
#             input_ids=input_ids,
#             attention_mask=encoder_attention_mask,
#         )['last_hidden_state']
#         encoder_outputs = torch.concatenate([encoder_outputs, torch.zeros_like(encoder_outputs)], dim=0)
#         encoder_attention_mask = torch.concatenate(
#                     [encoder_attention_mask, torch.zeros_like(encoder_attention_mask)], dim=0
#                 )
#         return encoder_outputs, encoder_attention_mask

#     def apply_delay_pattern_mask(self, input_ids, decoder_pad_token_mask):
#         """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
#         the mask is set to -1, and otherwise setting to the value detailed in the mask."""
#         seq_len = input_ids.shape[-1]
#         decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
#         input_ids = torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
#         return input_ids

#     def prepare_inputs_for_generation(self, input_ids, encoder_outputs, delay_pattern_mask):
#         input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
#         # for classifier free guidance we need to replicate the decoder args across the batch dim (we'll split these
#         # before sampling)
#         input_ids = input_ids.repeat((2, 1))
#         return input_ids, encoder_outputs

#     def prepare_decoder_input_ids_for_generation(self, batch_size: int):
#         return torch.ones(batch_size * self.num_codebooks, 1, dtype=torch.long) * self.decoder_start_token_id

#     def preprocess(self, text_ids: torch.Tensor, max_length: int)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#         with torch.no_grad():
#             batch_size = text_ids.shape[0]
#             encoder_outputs, encoder_attention_mask = self.prepare_text_encoder_kwargs_for_generation(text_ids)
#             encoder_outputs = self.enc_to_dec_proj(encoder_outputs)
#             input_ids = self.prepare_decoder_input_ids_for_generation(batch_size)
#             input_ids, delay_pattern_mask = self.delay_mask_fn(input_ids, self.decoder_start_token_id, max_length, self.num_codebooks, self.audio_channels)
#         return input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask

#     def sample_step(self, input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask):
#         i_ids, enc_out = self.prepare_inputs_for_generation(input_ids, encoder_outputs, delay_pattern_mask)
#         outputs = self.lm(input_ids=i_ids, encoder_hidden_states=enc_out, encoder_attention_mask=encoder_attention_mask)
#         next_token_logits = outputs['logits'][:, -1, :]
#         # TODO temperature
#         next_token_scores = self.logits_processor(input_ids, next_token_logits)
#         probs = nn.functional.softmax(next_token_scores, dim=-1)
#         next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
#         # update generated ids, model inputs, and length for next step
#         input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
#         return input_ids # update input_ids in the next call

#     def postprocess(self, input_ids: torch.Tensor, delay_pattern_mask: torch.Tensor, text_ids: torch.Tensor):
#         batch_size = text_ids.shape[0]
#         output_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
#         output_ids = output_ids[output_ids != self.decoder_start_token_id].reshape(
#             batch_size, self.num_codebooks, -1
#         )
#         # append the frame dimension back to the audio codes
#         output_ids = output_ids[None, ...]
#         output_values = self.audio_decoder(output_ids)
#         return output_values # update input_ids in the next call

#     def forward(self, text_ids: torch.Tensor, max_length: int):
#         with torch.no_grad():
#             input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask = self.preprocess(text_ids, max_length)
#             # sample
#             for _ in range(max_length-1):
#                 input_ids = self.sample_step(input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask)
#             output_values = self.postprocess(input_ids, delay_pattern_mask, text_ids)
#         return output_values


class NonRealtimeMusicGenModelWrapper(NonRealtimeTokenizerBase):
    def get_model_name(self) -> str:
        return "MusicGen"

    def get_model_authors(self) -> List[str]:
        return ["Naotake Masuda"]

    def get_model_short_description(self) -> str:
        return ""

    def get_model_long_description(self) -> str:
        return ""

    def get_technical_description(self) -> str:
        return ""

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/neutone_gen/example_clipper_gen.py"
        }

    def get_tags(self) -> List[str]:
        return ["clipper"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return False

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            DiscreteTokensNeutoneParameter(
                "texttokens",
                "tokens from a text tokenizer",
                default_value=[
                    2775,
                    7,
                    2783,
                    1463,
                    28,
                    7981,
                    63,
                    5253,
                    7,
                    11,
                    13353,
                    1,
                ],
            ),
            ContinuousNeutoneParameter(
                "outputlength", "number of output tokens", default_value=0.5
            ),
        ]

    @torch.jit.export
    def get_audio_in_channels(self) -> List[int]:
        return []

    @torch.jit.export
    def get_audio_out_channels(self) -> List[int]:
        return [1]

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [32000]

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return []  # Supports all buffer sizes

    @torch.jit.export
    def is_one_shot_model(self) -> bool:
        return True

    @torch.jit.export
    def has_progress_percentage(self) -> bool:
        return True

    def aggregate_continuous_params(self, cont_params: torch.Tensor) -> torch.Tensor:
        return cont_params  # We want sample-level control, so no aggregation

    def do_forward_pass(
        self,
        curr_block_idx: int,
        audio_in: List[torch.Tensor],
        knob_params: Dict[str, torch.Tensor],
        text_params: List[str],
        tokens_params: List[List[int]],
    ) -> List[torch.Tensor]:
        audio_out = []
        output_length = int(knob_params["outputlength"].mean() * 500)
        tokens = tokens_params[0]
        # Convert to LongTensor with batch size of 1
        tokens = torch.LongTensor(tokens).unsqueeze(0)
        with torch.no_grad():
            input_ids, encoder_outputs, delay_pattern_mask, encoder_attention_mask = (
                self.model.preprocess(tokens, output_length)
            )
            for i in range(output_length - 1):
                input_ids = self.model.sample_step(
                    input_ids,
                    encoder_outputs,
                    delay_pattern_mask,
                    encoder_attention_mask,
                )
                self.set_progress_percentage(float(i + 1) / output_length * 100)
                if self.should_cancel_forward_pass():
                    # Can't return empty list for some reason
                    break
            x = self.model.postprocess(input_ids, delay_pattern_mask, tokens)
        audio_out.append(x.squeeze(1))
        return audio_out
        # return [self.model.forward(min_val, min_val, max_val, gain)]


if __name__ == "__main__":
    from tokenizers import Tokenizer, SentencePieceUnigramTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="musicgen-model", type=str)
    args = parser.parse_args()
    model = torch.jit.load("../../out/musicgen_scripted_notok.ts")
    if TOK_TYPE == TokenizerType.SENTENCEPIECE:
        tok_path = str("../../out/spiece.model")
        with open(tok_path, mode="rb") as f:
            tok_string = base64.b64encode(f.read()).decode()
        tokenizer = SentencePieceUnigramTokenizer.from_spm(tok_path)
    elif TOK_TYPE == TokenizerType.JSON:
        tok_path = str("../../out/tokenizer.json")
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_string = json.dumps(json.load(f), ensure_ascii=True)
        tokenizer = Tokenizer.from_file(tok_path)

    wrapped = NonRealtimeMusicGenModelWrapper(model, tok_string, TOK_TYPE)
    tokens = tokenizer.encode("80s pop track with bassy drums and synth").ids

    sqw = NonRealtimeSampleQueueWrapper(wrapped)
    out = sqw.forward_non_realtime(
        [],
        torch.ones(1, 2048) * 0.2,
        tokens_params=[tokens],
    )
    sqw.reset()
    sqw.prepare_for_inference()
    log.info(f"   out[0].shape: {out[0].shape}")
    log.info(f"   out: {out}")
    ts = torch.jit.script(sqw)
    log.info(f"Scripting successful")
    n_samples = 2048
    tokens = tokenizer.encode("90s rock song with loud guitars and heavy drums").ids
    out_ts = ts.forward_non_realtime(
        [],
        torch.ones(1, 2048) * 0.2,
        tokens_params=[tokens],
    )
    log.info(f"out_ts[0].shape: {out_ts[0].shape}")
    log.info(f"out_ts: {out_ts}")
    torchaudio.save("../../out/out_ts.wav", out_ts[0], sample_rate=32000)
    torch.jit.save(ts, "../../out/wrapped-musicgen.ts")
    model = torch.jit.load("../../out/wrapped-musicgen.ts")
    # test saved tokenizer
    print(f"saved with {model.get_tokenizer_type()} tokenizer")
    if TOK_TYPE == TokenizerType.SENTENCEPIECE:
        tok_bin = base64.b64decode(model.get_tokenizer_str())
        # Create a named temporary file that is deleted when closed
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".model"
        ) as temp_model_file:
            temp_model_file.write(tok_bin)
            temp_model_file_path = temp_model_file.name
            tokenizer = SentencePieceUnigramTokenizer.from_spm(temp_model_file_path)
    elif TOK_TYPE == TokenizerType.JSON:
        tokenizer = Tokenizer.from_str(model.get_tokenizer_str())
    print(tokenizer.decode(tokens))
