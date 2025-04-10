# SPDX-License-Identifier: Apache-2.0

"""
This example shows how to infer or call a model using the framework and an OpenAI
backend to serve the model.

It uses MBRD and ROUGE scoring for majority voting to decide on best answer to use
from number of model sample outputs.
"""

# Local
from granite_io import make_backend, make_io_processor
from granite_io.io.voting import MBRDMajorityVotingProcessor
from granite_io.types import ChatCompletionInputs, UserMessage

# By default the OpenAI backend is configured for Ollama server running locally using
# default url. Ollama howevere does not support multiple completions per
# prompt i.e. "n" param below. You will need to use a backend that supports multiple
# completions to fully see majority voting happening. Backends like vLLM support
# multiple completions. You will need to update url and key when connecting to another
# backend.
model_name = "granite3.2:8b"
# openai_base_url = ""
# openai_api_key = ""
base_processor = make_io_processor(
    model_name,
    backend=make_backend(
        "openai",
        {
            "model_name": model_name,
            # "openai_base_url": openai_base_url,
            # "openai_api_key": openai_api_key,
        },
    ),
)
question = "What is 234651 + 13425?\nAnswer with just a number please."
messages = [UserMessage(content=question)]
completion_inputs = ChatCompletionInputs(
    messages=messages,
    thinking=True,
    generate_inputs={"n": 20, "temperature": 0.6, "max_tokens": 1024},
)
voting_processor = MBRDMajorityVotingProcessor(base_processor)
results = voting_processor.create_chat_completion(completion_inputs)
print("Output from base model augmented with MBRD majority voting:")
# This should be only one output, the majority voted answer
for result_num, r in enumerate(results.results):
    print(f"{result_num + 1}: {r.next_message.content}")

# What's the actual answer?
print(f"---------\nThe actual answer is: {234651 + 13425}")
