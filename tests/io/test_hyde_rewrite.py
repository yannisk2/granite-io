# SPDX-License-Identifier: Apache-2.0

"""
Test cases for query_rewrite.py
"""

# Standard
import datetime
import textwrap

# Local
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.hyde_rewrite import HyDERewriteIOProcessor
from granite_io.types import (
    GenerateResult,
    GenerateResults,
)

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {"role": "assistant", "content": "Welcome to pet questions!"},
            {
                "role": "user",
                "content": "Hi!",
            },
            {
                "role": "assistant",
                "content": "Hello, how can I assist you today?",
            },
            {"role": "user", "content": "I have some questions about safe room"},
            {
                "role": "assistant",
                "content": "Sure, I can help with that. What would you like to know?",
            },
            {
                "role": "user",
                "content": "What items I need to keep in the safe room?",
            },
        ],
        "generate_inputs": {
            "temperature": 0.0  # Ensure consistency across runs
        },
    }
)


def _make_result(content: str):
    """Convenience method to create a fake model output object."""
    return GenerateResult(
        completion_string=content, completion_tokens=[], stop_reason="dummy stop reason"
    )


_TODAYS_DATE = datetime.datetime.now().strftime("%B %d, %Y")

HYDE_INSTRUCTION = (
    "Generate a short answer for the user query below. Do not use more than 50 words.\n"
)


def test_canned_input():
    """
    Validate that the I/O processor handles a single instance of canned input in the
    expected way.
    """
    io_processor = HyDERewriteIOProcessor(None)
    output = io_processor.inputs_to_generate_inputs(_EXAMPLE_CHAT_INPUT)

    expected_output = textwrap.dedent(f"""<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
Today's Date: {_TODAYS_DATE}.
You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>{HYDE_INSTRUCTION}What items I need to keep in the safe room?<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>""")  # noqa: E501

    print(f"---\n|{output.prompt}|\n---\n")
    print(f"|{expected_output}|\n---")

    assert output.prompt == expected_output


def test_canned_output():
    """
    Validate that the I/O processor handles a single instance of canned model output
    in the expected way.
    """
    io_processor = HyDERewriteIOProcessor(None)

    raw_output_to_expected = [
        (
            "What is the capital of France",  # noqa: E501
            "Is Rex more likely to get fleas because he spends a lot of time outdoors?",
        ),
        # Current code passes through malformed data unchanged
        ("<invalid model response>", "<invalid model response>"),
    ]

    # Single output
    for raw_output, expected in raw_output_to_expected:
        output = io_processor.output_to_result(
            GenerateResults(results=[_make_result(raw_output)]), _EXAMPLE_CHAT_INPUT
        )
        print(f"\n{output}\n")

        assert len(output.results) == 1
        assert output.results[0].next_message.content == expected

    # Multiple outputs
    multi_raw_output = [
        _make_result(raw_output) for raw_output, _ in raw_output_to_expected
    ]
    multi_expected = [expected for _, expected in raw_output_to_expected]
    multi_output = io_processor.output_to_result(
        GenerateResults(results=multi_raw_output), _EXAMPLE_CHAT_INPUT
    )
    multi_output_strs = [r.next_message.content for r in multi_output.results]
    print(f"\n{output}\n")
    assert multi_output_strs == multi_expected
