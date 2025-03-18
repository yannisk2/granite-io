# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import os

# Local
from granite_io import get_input_processor
from granite_io.types import (
    ChatCompletionInputs,
    UserMessage,
)

_GENERALE_MODEL_NAME = "Granite 3.2"
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def _load_model_output_file(file_name: str) -> str:
    response = Path(file_name).read_text(encoding="UTF-8")
    return response


def test_run_processor_reasoning():
    input_processor = get_input_processor(_GENERALE_MODEL_NAME)
    question = (
        "Find the fastest way for a seller to visit all the cities in their region"
    )
    messages = [UserMessage(content=question)]
    prompt = input_processor.transform(
        ChatCompletionInputs(messages=messages, thinking=True)
    )

    expected_prompt = _load_model_output_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_prompt.txt")
    )
    assert isinstance(prompt, str)
    assert prompt == expected_prompt
