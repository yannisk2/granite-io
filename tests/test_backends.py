# SPDX-License-Identifier: Apache-2.0

# Standard
import re

# Third Party
from litellm import UnsupportedParamsError
import pytest

# Local
from granite_io.backend.openai import OpenAIBackend
from granite_io.types import GenerateInputs, GenerateResults


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_simple(backend_x):
    ret = await backend_x(GenerateInputs(prompt="hello"))
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_num_return_sequences_1(backend_x):
    ret = await backend_x(GenerateInputs(prompt="hello", n=1))
    assert isinstance(ret, GenerateResults)
    assert len(ret.results) == 1


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_num_return_sequences_3(backend_x):
    try:
        ret = await backend_x(GenerateInputs(prompt="what is up?", n=3))
    except UnsupportedParamsError:
        # Specific exception from LiteLLMBackend
        # xfail because LiteLLM is telling us that ollama does not support
        # n > 1, but we can use LiteLLM with watsonx FTW
        pytest.xfail("LiteLLMBackend support for n > 1 varies by provider")

    num_returned = len(ret.results)

    if num_returned == 1 and isinstance(backend_x, OpenAIBackend):
        # ollama with OpenAI will just return 1, other OpenAI backends can return 3
        pytest.xfail("OpenAIBackend support for n > 1 varies by provider")

    assert isinstance(ret, GenerateResults)
    assert num_returned == 3


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("n", [-1, 0])
async def test_num_return_sequences_invalid(backend_x, n):
    with pytest.raises(ValueError, match=re.escape(f"Invalid value for n ({n})")):
        _ = await backend_x(inputs=GenerateInputs(prompt="hello", n=n))
