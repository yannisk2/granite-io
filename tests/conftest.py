# SPDX-License-Identifier: Apache-2.0

# Standard
import collections.abc

# Third Party
import pytest
import torch

# Local
from granite_io import make_backend
from granite_io.backend import Backend
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.consts import (
    _GRANITE_3_2_2B_HF,
    _GRANITE_3_3_2B_HF,
    _GRANITE_3_3_2B_OLLAMA,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    override_date_for_testing as g32_override_date_for_testing,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    override_date_for_testing as g33_override_date_for_testing,
)
from granite_io.io.rag_agent_lib import obtain_lora


@pytest.fixture(scope="session")
def vcr_config():
    return {"filter_headers": ["authorization"]}


def backend_openai() -> Backend:
    return make_backend(
        "openai",
        {
            "model_name": "granite3.2:2b",
            "openai_api_key": "ollama",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )


def backend_litellm() -> Backend:
    return make_backend(
        "litellm",
        {
            "model_name": "ollama/" + "granite3.2:2b",
        },
    )


def backend_transformers() -> Backend:
    return make_backend(
        "transformers",
        {
            "model_name": _GRANITE_3_2_2B_HF,
        },
    )


def backend_3_3_openai() -> Backend:
    return make_backend(
        "openai",
        {
            "model_name": _GRANITE_3_3_2B_OLLAMA,
            "openai_api_key": "ollama",
            "openai_base_url": "http://localhost:11434/v1",
        },
    )


def backend_3_3_litellm() -> Backend:
    return make_backend(
        "litellm",
        {
            "model_name": "ollama/" + _GRANITE_3_3_2B_OLLAMA,
        },
    )


def backend_3_3_transformers() -> Backend:
    return make_backend(
        "transformers",
        {
            "model_name": _GRANITE_3_3_2B_HF,
        },
    )


@pytest.fixture(scope="function")
def fake_date():
    """
    We need today's date to be constant so that ``vcrpy`` functions on our prompts.

    By wrapping the creation of this date in a fixture, we can be sure to reset to
    normal behavior if a test fails.

    :returns: a fake version of today's date, for use in prompt text that mentions the
     date.
    """
    yield "April 1, 2025"

    # Cleanup code. Augment as needed as we add new IO processors with date-dependent
    # prompts.
    g32_override_date_for_testing(None)
    g33_override_date_for_testing(None)


@pytest.fixture(
    scope="session", params=[backend_openai, backend_litellm, backend_transformers]
)
def backend_x(request) -> Backend:
    return request.param()


@pytest.fixture(
    scope="session",
    params=[backend_3_3_openai, backend_3_3_litellm, backend_3_3_transformers],
)
def backend_3_3(request) -> Backend:
    return request.param()


@pytest.fixture(scope="session")
def lora_server() -> collections.abc.Generator[LocalVLLMServer, object, None]:
    """
    Fixture that runs a local vLLM server. The server uses a fixed port because the
    ``vcrpy`` package requires fixed local ports.

    :returns: vLLM server with all the LoRAs for which we currently have IO processors

    """
    if not torch.cuda.is_available():
        pytest.xfail("GPU required to run vLLM. vLLM required to run model.")

    # Updated to use Granite 3.3 8B with latest LoRA adapters
    base_model = "ibm-granite/granite-3.3-8b-instruct"

    # LoRA adapter short names - these will be resolved to local paths using
    # obtain_lora()
    lora_adapter_names = [
        "answerability_prediction",  # Maps to answerability_prediction_lora
        "certainty",  # Maps to certainty_lora
        "citation_generation",  # Maps to citation_generation_lora
        "hallucination_detection",  # Maps to hallucination_detection_lora
        "query_rewrite",  # Maps to query_rewrite_lora
        "context_relevancy",  # Maps to context_relevancy_lora
    ]

    # Download and get local paths for all LoRA adapters
    lora_adapters = []
    for lora_name in lora_adapter_names:
        try:
            lora_path = obtain_lora(lora_name)
            lora_adapters.append((lora_name, str(lora_path)))
            print(f"✅ Downloaded LoRA adapter: {lora_name} -> {lora_path}")
        except (OSError, ValueError, RuntimeError) as e:
            print(f"❌ Failed to download LoRA adapter {lora_name}: {e}")
            # Continue with other adapters

    server = LocalVLLMServer(base_model, lora_adapters=lora_adapters, port=35782)
    server.wait_for_startup(200)
    yield server

    # Shutdown code runs at end of test session
    server.shutdown()
