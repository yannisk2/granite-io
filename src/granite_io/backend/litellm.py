# SPDX-License-Identifier: Apache-2.0


# Standard
from typing import TYPE_CHECKING

# Local
from granite_io.backend.base import Backend
from granite_io.backend.registry import backend
from granite_io.optional import import_optional
from granite_io.types import GenerateInputs

if TYPE_CHECKING:
    # Third Party
    import litellm  # noqa: F401


@backend(
    "litellm",
    config_schema={
        "properties": {
            "model_name": {"type": "string"},
            # "openai_base_url": {"type": "string"},
            # "openai_api_key": {"type": "string"},
        }
    },
)
class LiteLLMBackend(Backend):
    async def generate(self, inputs: GenerateInputs):
        """Run a direct /completions call"""

        with import_optional("litellm"):
            # Third Party
            import litellm

        return await litellm.atext_completion(**inputs.model_dump())
