# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite query expansion intrinsic.
"""

# Standard
import json

# Third Party
import pydantic

# Local
from granite_io.io.base import (
    ModelDirectInputOutputProcessorWithGenerate,
    InputOutputProcessor,
)
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2InputProcessor,
    Granite3Point2Inputs,
)
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
    UserMessage,
)

import asyncio

class QueryExpansionIOProcessor(InputOutputProcessor):
    def __init__(
        self,
        backend,
        io_processor,
        rewrite_request_proc,
    ):
        self.backend = backend
        self.io_proc = io_processor
        self.rewrite_request_proc = rewrite_request_proc
    
    @staticmethod
    def format_chat_history(messages):
        formatted = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)    

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        
        coroutines = []
        #################### Run QUERY REWRITE LoRA ####################
        coroutines.append(self.rewrite_request_proc.acreate_chat_completion(inputs))        
        ####################Reformulate into Synonymous Query by prompting Granite####################
        input_conversation_string = self.format_chat_history(inputs.model_dump()['messages'])
        # print(f"Input Conversation: {input_conversation_string}")
        
        generate_inputs = GenerateInputs(max_tokens=512, top_p=1, temperature=1, stop="[[Input]]")
        generate_inputs.prompt = f"""You are given a multi-turn conversation between a user and an assistant. Reformulate the last-turn user query into a synonymous standalone query by replacing key terms with appropriate synonyms or closely related phrases, while preserving the original intent and meaning. This rewritten query will be used to retrieve relevant passages from a corpus, so it must remain faithful to the user's information need. Only output the rewritten query.\n\n[[Input]]\n{input_conversation_string}\n\n[[Output]]\n"""
        coroutines.append(self.backend.pipeline(generate_inputs))
        ####################Query Enrichment by prompting Granite####################
        generate_inputs = GenerateInputs(max_tokens=512, top_p=1, temperature=1)
        generate_inputs.prompt = f"""Your task is to generate a list of short, highly relevant technical keywords or search queries based on the conversation history, focusing on the user's last query. These keywords will be used to retrieve the most relevant technical passages. Use domain-specific terms, abbreviations, and key phrases where appropriate. Do not answer the question.\n\n[[Conversation]]\n{input_conversation_string}\n\n[[Search Keywords]]"""
        coroutines.append(self.backend.pipeline(generate_inputs))
        #################### Sample Answer from Granite ####################
        chat_input_ans_v1 = ChatCompletionInputs(messages=inputs.messages, generate_inputs={"temperature": 1, "top_p": 1, "max_tokens": 512, })
        coroutines.append(self.io_proc.acreate_chat_completion(chat_input_ans_v1))
        ################################################################################
        
        # Merge results from parallel invocations
        sub_results = await asyncio.gather(*coroutines)
        # print("sub_results", sub_results)
        print("QUERY REWRITE:", sub_results[0])
        print("SYNONMOUS QUERY:", sub_results[1])
        print("ENRICHED QUERIES:", sub_results[2])
        print("SAMPLED ANSWER:", sub_results[3])
        
        
        query_answer_v1 = sub_results[3].results[0].next_message.content
        print("query_answer_v1", query_answer_v1)
        ####################Reverse-Engineer the Question from the Sampled Answer####################
        generate_inputs = GenerateInputs(max_tokens=512, top_p=1, temperature=1, stop="[[Answer]]")
        generate_inputs.prompt = f"Generate a single question for the given answer.\n[[Answer]]\nAlbert Einstein was born in Germany.\n[[Question]]\nWhere was Albert Einstein born?\n[[Answer]]{query_answer_v1}\n[[Question]]\n"

        RevQ_v1_output = await self.backend.pipeline(generate_inputs)
        print(f"Reverse-Engineered Question: {RevQ_v1_output}")
        
        last_user_message = inputs.model_dump()['messages'][-1]
        query_str_list = [
            last_user_message['content'],                       #0
            sub_results[0].results[0].next_message.content,     #1
            sub_results[1].results[0].completion_string,        #2
            sub_results[2].results[0].completion_string,        #3
            sub_results[3].results[0].next_message.content,     #4
            RevQ_v1_output.results[0].completion_string,        #5
        ]
        
        # print("\nList of Query Strings:", query_str_list)    
        
        results = []
        for cur_query in query_str_list:
            results.append(ChatCompletionResult(next_message=UserMessage(content=cur_query)))
        # print("results", results)
        
        return ChatCompletionResults(results=results)