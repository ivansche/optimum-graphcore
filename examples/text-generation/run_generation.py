#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
import sys

import numpy as np
import torch

import poptorch
from optimum.graphcore import IPUConfig
from optimum.graphcore.ipu_configuration import ALLOWED_POD_TYPES
from optimum.graphcore.modeling_utils import to_pipelined
from transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2-medium": (GPT2LMHeadModel, GPT2Tokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def adjust_length_to_model(ipu, length, max_sequence_length, prompt_length):
    if ipu:
        # TODO: assert apply tp CPU too?
        assert prompt_length < max_sequence_length, 'Prompt length must be smaller than max sequence length'
        # Note that length + prompt_length must <= max_sequence_length
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length - prompt_length
        elif 0 < max_sequence_length < length + prompt_length:
            length = max_sequence_length - prompt_length # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length
    else:
        if length < 0 and max_sequence_length > 0:
            length = max_sequence_length
        elif 0 < max_sequence_length < length:
            length = max_sequence_length  # No generation bigger than model size
        elif length < 0:
            length = MAX_LENGTH  # avoid infinite loop
        return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--ipu_config_name",
        default=None,
        type=str,
        help="Pretrained IPU config name or path if not the same as model_name.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Path to directory to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--ipu_config_overrides",
        default=None,
        type=str,
        help="Override some existing ipu config settings. Example: device_iterations=4,gradient_accumulation_steps=64",
    )
    parser.add_argument(
        "--pod_type",
        default=None,
        type=str,
        help="The POD type to run the `Trainer` on. Choices:" + ", ".join(ALLOWED_POD_TYPES),
    )

    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20, help="The length of the number of generated tokens.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length (prompt length + generation length)",
    )
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--ipu", action="store_true", help="Use IPUs")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    if args.ipu:
        ipu_config = IPUConfig.from_pretrained(
            args.ipu_config_name if args.ipu_config_name else args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        ipu_config = ipu_config.for_pod_type(args.pod_type)
        if args.ipu_config_overrides:
            logger.info(f"Overriding IPU config: {args.ipu_config_overrides}")
            ipu_config.update_from_string(args.ipu_config_overrides)
        ipu_config = ipu_config.for_pod_type(args.pod_type)
        logger.info(ipu_config)

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    if args.ipu:
        args.device = "ipu"
        model = to_pipelined(model, ipu_config, force=False)
        model.parallelize()
        if args.fp16:
            model.half()
        opts = ipu_config.to_options(for_inference=True)
        # TODO: Should use pipelineing when doing auto-regressive generation
        opts.setExecutionStrategy(poptorch.ShardedExecution())
        model = poptorch.inferenceModel(model, opts)
        model.eval()
        model._user_model.ipu_executor = model
        model._user_model.max_seq_length = args.max_seq_length
    else:
        args.device = "cpu"
        model.to("cpu")

    logger.warning(f"device: {args.device}, 16-bits training: {args.fp16}")

    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    prefix = args.prefix if args.prefix else args.padding_text
    encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    # encoded_prompt = encoded_prompt.to(args.device)

    args.length = adjust_length_to_model(
        args.ipu, 
        args.length, 
        max_sequence_length=args.max_seq_length if args.max_seq_length else model.config.max_position_embeddings, 
        prompt_length=len(encoded_prompt[0])
    )
    logger.info(f"Length after adjustment: {args.length}")

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        # Print utf-8 output
        sys.stdout.buffer.write(total_sequence.encode('utf-8'))

    return generated_sequences


if __name__ == "__main__":
    main()
