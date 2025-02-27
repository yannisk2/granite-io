# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
from openai import OpenAI
import aconfig

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.types import GenerateResult


@backend(
    "openai",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
            "openai_base_url": {"type": "string"},
            "openai_api_key": {"type": "string"},
        }
    },
)
class OpenAIBackend(Backend):
    _model_str: str
    _openai_client: OpenAI

    def __init__(self, config: aconfig.Config):
        self._model_str = config.model_name
        api_key = config.get("openai_api_key", "ollama")
        base_url = config.get("openai_base_url", "http://localhost:11434/v1")

        default_headers = {"RITS_API_KEY": api_key} if api_key else None

        self.openai_client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
        )

    def generate(self, input_str: str) -> GenerateResult:
        """Run a direct /completions call"""

        result = self.openai_client.completions.create(
            model=self._model_str,
            prompt=input_str,
        )
        return GenerateResult(
            completion_string=result.choices[0].text,
            completion_tokens=[],  # Not part of the OpenAI spec
            stop_reason=result.choices[0].finish_reason,
        )
