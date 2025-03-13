# SPDX-License-Identifier: Apache-2.0

# Local
from granite_io.io.base import OutputProcessor
from granite_io.io.consts import (
    _GRANITE_3_2_COT_END,
    _GRANITE_3_2_COT_START,
)
from granite_io.io.output_processors.granite_3_2.granite_output_parser import (
    parse_model_output,
)
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateResults,
)

# Some versions of the model are known to shorten "Here is" to "Here's", so we
# provide alternate forms of these strings for those versions.
_COT_START_ALTERNATIVES = [
    _GRANITE_3_2_COT_START,
    "Here's my thought process:",
]
_COT_END_ALTERNATIVES = [
    _GRANITE_3_2_COT_END,
    "Here's my response:",
]


class Granite3Point2OutputProcessor(OutputProcessor):
    """
    Output processor for version 3.2 of the main Granite models, all sizes.

    This Output processor is based on the Jinja template that was used during
    supervised fine tuning of these models. This template is as follows:
    ```
    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content'] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = \"Knowledge Cutoff Date: April 2024.\nToday's Date: \"
          + strftime_now('%B %d, %Y') + \".\nYou are Granite, developed by IBM.\" %}
        {%- if tools and documents %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools.
                  When a tool is required to answer the user's query, respond with
                  <|tool_call|> followed by a JSON list of tools used. If a tool does
                  not exist in the provided list of tools, notify the user that you do
                  not have the ability to fulfill the request.\n\nWrite the response to
                  the user's input by strictly aligning with the facts in the provided
                  documents. If the information needed to answer the question is not
                  available in the documents, inform the user that the question cannot
                  be answered based on the available data.\" %}
        {%- elif tools %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools. When a tool is required to
                answer the user's query, respond with <|tool_call|> followed by a JSON
                list of tools used. If a tool does not exist in the provided list of
                tools, notify the user that you do not have the ability to fulfill the
                request.\" %}
        {%- elif documents %}
                {%- set system_message = system_message + \" Write the response to the
                user's input by strictly aligning with the facts in the provided
                documents. If the information needed to answer the question is not
                available in the documents, inform the user that the question cannot be
                answered based on the available data.\" %}
        {%- elif thinking %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\nRespond to every user query in a comprehensive and detailed
                way. You can write down your thoughts and reasoning process before
                responding. In the thought process, engage in a comprehensive cycle of
                analysis, summarization, exploration, reassessment, reflection,
                backtracing, and iteration to develop well-considered thinking process.
                In the response section, based on various attempts, explorations, and
                reflections from the thoughts section, systematically present the final
                solution that you deem correct. The response should summarize the
                thought process. Write your thoughts after 'Here is my thought process:'
                and write your response after 'Here is my response:' for each user
                query.\" %}
        {%- else %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\" %}
        {%- endif %}
        {%- if 'citations' in controls and documents %}
            {%- set system_message = system_message + '\n\nIn your response, use the
            symbols <co> and </co> to indicate when a fact comes from a document in the
            search result, e.g <co>0</co> for a fact from document 0. Afterwards, list
            all the citations with their corresponding documents in an ordered list.' %}
        {%- endif %}
        {%- if 'hallucinations' in controls and documents %}
            {%- set system_message = system_message + '\n\nFinally, after the response
            is written, include a numbered list of sentences from the response that are
            potentially hallucinated and not based in the documents.' %}
        {%- endif %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {{- '<|start_of_role|>system<|end_of_role|>' + system_message +
        '<|end_of_text|>\n' }}
    {%- if tools %}
        {{- '<|start_of_role|>tools<|end_of_role|>' }}
        {{- tools | tojson(indent=4) }}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- if documents %}
        {{- '<|start_of_role|>documents<|end_of_role|>' }}
        {%- for document in documents %}
            {{- 'Document ' + loop.index0 | string + '\n' }}
            {{- document['text'] }}
            {%- if not loop.last %}
                {{- '\n\n'}}
            {%- endif%}
        {%- endfor %}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- for message in loop_messages %}
        {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' +
        message['content'] + '<|end_of_text|>\n' }}
        {%- if loop.last and add_generation_prompt %}
            {{- '<|start_of_role|>assistant' }}
            {%- if controls %}
                {{- ' ' + controls | tojson()}}
            {%- endif %}
            {{- '<|end_of_role|>' }}
        {%- endif %}
    {%- endfor %}
    ```
    """

    def transform(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,
    ) -> ChatCompletionResults:
        results = []
        for result in output.results:
            output = result.completion_string
            original_output = output

            # Parse out CoT reasoning
            cot = None
            if inputs.thinking:
                cot_start_span = None
                cot_end_span = None
                for cot_start_str in _COT_START_ALTERNATIVES:
                    if (cot_start_pos := output.find(cot_start_str)) != -1:
                        cot_start_span = (
                            cot_start_pos,
                            cot_start_pos + len(cot_start_str),
                        )
                        break
                for cot_end_str in _COT_END_ALTERNATIVES:
                    if (cot_end_pos := output.find(cot_end_str)) != -1:
                        cot_end_span = (cot_end_pos, cot_end_pos + len(cot_end_str))
                        break

                if (
                    cot_start_span
                    and cot_end_span
                    and cot_end_span[0] > cot_start_span[1]
                ):
                    cot = output[cot_start_span[1] : cot_end_span[0]].strip()
                    output = (
                        output[: cot_start_span[0]] + output[cot_end_span[1] :].strip()
                    )

            # Parse out tool calls
            if output.startswith("<tool_call>"):
                raise NotImplementedError("TODO: Implement tool call parsing")

            # Parse out citations, documents and hallucinations
            try:
                parsed_output = parse_model_output(output, inputs.documents)
            except Exception as err:
                raise ValueError(
                    "Failed to parse citations, documents and hallucinations "
                    "from model ouput."
                ) from err

            results.append(
                ChatCompletionResult(
                    next_message=AssistantMessage(
                        citations=parsed_output["citations"],
                        content=parsed_output["response"],
                        documents=parsed_output["docs"],
                        hallucinations=parsed_output["hallucinations"],
                        reasoning_content=cot,
                        raw=original_output,
                    )
                )
            )

        return ChatCompletionResults(results=results)
