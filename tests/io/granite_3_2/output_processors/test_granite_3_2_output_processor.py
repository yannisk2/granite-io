# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import os

# Local
from granite_io import get_output_processor
from granite_io.types import (
    ChatCompletionInputs,
    GenerateResult,
    GenerateResults,
    UserMessage,
)

_GENERAL_MODEL_NAME = "Granite 3.2"
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _load_model_output_file(file_name: str) -> str:
    response = Path(file_name).read_text(encoding="UTF-8")
    return response


def test_run_processor_reasoning():
    output_processor = get_output_processor(_GENERAL_MODEL_NAME)
    # Process the output
    results = []
    raw_output = _load_model_output_file(
        os.path.join(_TEST_DATA_DIR, "test_raw_reasoning_output.txt")
    )
    results.append(
        GenerateResult(
            completion_string=raw_output,
            completion_tokens=[],
            stop_reason="stop",
        )
    )
    question = (
        "Find the fastest way for a seller to visit all the cities in their region"
    )
    messages = [UserMessage(content=question)]
    outputs = output_processor.transform(
        GenerateResults(results=results),
        ChatCompletionInputs(messages=messages, thinking=True),
    )

    expected_thought_output = _load_model_output_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_output_processor_thought.txt")
    )
    assert expected_thought_output == outputs.results[0].next_message.reasoning_content

    expected_response_output = _load_model_output_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_output_processor_response.txt")
    )
    assert expected_response_output == outputs.results[0].next_message.content
