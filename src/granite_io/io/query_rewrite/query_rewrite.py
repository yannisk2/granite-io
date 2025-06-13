# SPDX-License-Identifier: Apache-2.0

import json
import re
import pydantic

from granite_io.io.base import ModelDirectInputOutputProcessorWithGenerate
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)

# Regex for v3.3 constrained decoding
_JSON_FENCE_REGEX = r"```json\n\{\s*\"rewritten_question\"\s*:\s*\"[^\"]*\"\s*\}\n```"

# JSON object example for Granite 3.3, dumped with exact spacing
_JSON_OBJECT_3_3 = {
    "rewritten_question": "YOUR_REWRITTEN_QUESTION_HERE"
}

JSON_OBJECT_3_3_STR = json.dumps(_JSON_OBJECT_3_3, indent=4)

JSON_TEMPLATE_3_3 = (
    "Your output should be a JSON structure with the rewritten question:\n"
    "```json\n"
    f"{JSON_OBJECT_3_3_STR}\n"
    "```"
)

# Prompt templates
QUERY_TO_REWRITE_TEMPLATE = (
    "<|start_of_role|>query_to_rewrite<|end_of_role|>{msg}<|end_of_text|>\n"
)

INSTRUCTION_TEXT_3_3 = (
    "Given the conversation history above and the specific query provided in the "
    "'query_to_rewrite' role, rewrite that query into a standalone question that "
    "captures the user's intent without requiring the conversation context. "
    "If the query is already clear and standalone, output it as is."
)

REWRITE_PROMPT_3_3 = (
    "<|start_of_role|>rewrite: "
    + INSTRUCTION_TEXT_3_3
    + "\n"
    + JSON_TEMPLATE_3_3
    + "<|end_of_role|>"
)

REWRITE_PROMPT_3_2 = (
    "<|start_of_role|>rewrite: "
    "Reword the final utterance from the USER into a single utterance that doesn't "
    "need the prior conversation history to understand the user's intent. If the final "
    "utterance is a clear and standalone question, please DO NOT attempt to rewrite "
    "it, rather output the last user utterance as is."
    " Your output format should be in JSON: { \"rewritten_question\": <REWRITE> }"
    "<|end_of_role|>"
)

# Pydantic model for parsing raw JSON
class QueryRewriteRawOutput(pydantic.BaseModel):
    rewritten_question: str

RAW_OUTPUT_JSON_SCHEMA = QueryRewriteRawOutput.model_json_schema()


class QueryRewriteIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    Single-class I/O processor supporting both Granite 3.2 and 3.3 query-rewrite LoRAs.
    Pass `version="3.2"` or `version="3.3"` at init to switch behavior.
    """

    def __init__(self, backend, version: str = "3.3"):
        super().__init__(backend=backend)
        self.version = version

        if version == "3.3":
            from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
                Granite3Point3InputProcessor,
                Granite3Point3Inputs,
            )
            self.InputsModel = Granite3Point3Inputs
            self.base_input_processor = Granite3Point3InputProcessor()
            self._make_prompt = self._make_prompt_3_3
            self._extra_body = {"guided_regex": _JSON_FENCE_REGEX}
        elif version == "3.2":
            from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
                Granite3Point2InputProcessor,
                Granite3Point2Inputs,
            )
            self.InputsModel = Granite3Point2Inputs
            self.base_input_processor = Granite3Point2InputProcessor()
            self._make_prompt = self._make_prompt_3_2
            self._extra_body = {"guided_json": RAW_OUTPUT_JSON_SCHEMA}
        else:
            raise ValueError(f"Unsupported Granite version: {version}")

    def inputs_to_generate_inputs(
        self,
        inputs: ChatCompletionInputs,
        add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate and normalize inputs for the selected version
        inputs = self.InputsModel.model_validate(inputs.model_dump())

        if inputs.messages[-1].role != "user":
            raise ValueError("Last message is not a user message")

        # Build prompt prefix
        prompt_prefix = self.base_input_processor.transform(inputs, False)
        # Delegate to version-specific builder
        prompt = self._make_prompt(
            prompt_prefix,
            inputs.messages[-1].content,
            add_generation_prompt
        )
        
        return inputs.generate_inputs.model_copy(update={
            "prompt": prompt,
            "max_tokens": 256,
            "extra_body": self._extra_body,
        })

    def _make_prompt_3_3(self, prefix: str, final_msg: str, add_prompt: bool) -> str:
        prompt = prefix + QUERY_TO_REWRITE_TEMPLATE.format(msg=final_msg)
        if add_prompt:
            prompt += REWRITE_PROMPT_3_3
        return prompt

    def _make_prompt_3_2(self, prefix: str, final_msg: str, add_prompt: bool) -> str:
        prompt = prefix
        if add_prompt:
            prompt += REWRITE_PROMPT_3_2
        return prompt

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        results = []
        for raw in output.results:
            s = raw.completion_string.strip()
            rewrite = None

            # try JSON parse for both versions
            try:
                m = re.search(r"\{.*\}", s, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    rewrite = data.get("rewritten_question")
            except Exception:
                pass

            # fallback to raw string if parse failed
            if not rewrite:
                rewrite = s

            # replace last user message with the rewritten content
            rewritten = inputs.messages[-1].model_copy(update={"content": rewrite})
            results.append(ChatCompletionResult(next_message=rewritten))

        return ChatCompletionResults(results=results)