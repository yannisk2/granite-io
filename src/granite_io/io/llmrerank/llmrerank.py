# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for reranking retrieval output using Garnite model
"""

# Standard
import asyncio
import copy

# Local
from granite_io.io.base import InputOutputProcessor, RequestProcessor
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
)
from granite_io.types import ChatCompletionInputs

INSTRUCTION_TEXT = """You are a smart and helpful AI assistant with in-depth knowledge
about how people search for information using search engines. In this task, you are 
given two passages, and a query, your job is to judge which passage is relatively more
suitable to answer that query. The first passage will start with "passage A" and the 
second passage with start with "passage B". Output the preferred passage index, i.e. A 
or B and followed by an explanation, if none of the passage answer the query directly, 
pick the one that has more relevant information.
"""


def format_prompts(instruction, query, documents, generation_parameters):
    prompts = [
        f"{instruction}\n passage A: {pair[0]} \n passage B: {pair[1]} \n \
            query: {query}"
        for pair in documents
    ]
    prompts = [
        {"messages": [{"role": "user", "content": prompt}]} for prompt in prompts
    ]
    prompts = [prompt | generation_parameters for prompt in prompts]
    chat_input = [Granite3Point2Inputs.model_validate(prompt) for prompt in prompts]
    return chat_input


class RerankRequestProcessor(RequestProcessor):
    """
    Rerank processor that rerank the retrieval output by passing that output through
    an IO processor.
    """

    def __init__(
        self,
        io_proc: InputOutputProcessor,
        rerank_top_k: int = 8,
        return_top_k: int = 5,
        verbose: bool = False,
        prompt: str = None
    ):
        """
        :param io_proc: IO processor for a model that rerank the retrieval output
        :param rerank_top_k: Number of retrieved passages to rerank
        :param return_top_k: Number of reranked passages to return
        :param verbose: whether to generate explianation or not 
        :param prompt: user defined prompt
        """
        self._io_proc = io_proc
        self._rerank_top_k = rerank_top_k
        self._return_top_k = return_top_k
        self._verbose = verbose
        self._prompt = prompt if prompt else INSTRUCTION_TEXT
        self._generation_parameters = {
            "generate_inputs": {
                "temperature": 0.0,
                "max_tokens": 4096 if self._verbose else 2,
            },
            "do_sample": False,
        }

    async def aprocess(
        self, inputs: ChatCompletionInputs
    ) -> list[ChatCompletionInputs]:
        inputs_copy = copy.deepcopy(inputs)  # avoid changing original inputs
        query = inputs_copy.messages[-1].content
        documents = inputs_copy.documents
        if len(documents) % 2 != 0:
            print("for simplicity, drop the last document of number of document is odd")
        if self._rerank_top_k > len(documents):
            print(
                f"Number of document ({len(documents)}) is less than "
                f"{self._rerank_top_k} only reranking {len(documents)} documents"
            )
        total_tournament_passages = min(
            self._rerank_top_k, len(documents) - len(documents) % 2
        )

        while total_tournament_passages > 10:
            batch_content = []
            for indx in range(0, total_tournament_passages // 2):
                passage1 = documents[indx].text
                passage2 = documents[total_tournament_passages - 1 - indx].text
                content = [passage1, passage2]
                batch_content.append(content)
            chat_input = format_prompts(
                self._prompt, query, batch_content, self._generation_parameters
            )
            generations = await asyncio.gather(
                *[self._io_proc.acreate_chat_completion(c) for c in chat_input]
            )
            for indx in range(0, total_tournament_passages // 2):
                choice = generations[indx].results[0].next_message.content[0].upper()
                if choice == "B":
                    if self._verbose:
                        print("LLM prefers a lower ranked passage")
                        print(f"query: {query}")
                        print(f"passage A: {batch_content[indx][0]}")
                        print(f"passage A position: {indx}")
                        print(f"passage B: {batch_content[indx][1]}")
                        print(
                            f"passage B position: "
                            f"{total_tournament_passages - 1 - indx}"
                        )
                        print(
                            f"LLM output:"
                            f"{generations[indx].results[0].next_message.content}"
                        )
                    documents[indx], documents[total_tournament_passages - 1 - indx] = (
                        documents[total_tournament_passages - 1 - indx],
                        documents[indx],
                    )
            total_tournament_passages = total_tournament_passages // 2

        # all to all tournament
        batch_content = []
        pair_index = []
        for indx_a in range(0, total_tournament_passages):
            for indx_b in range(indx_a + 1, total_tournament_passages):
                batch_content.append([documents[indx_a].text, documents[indx_b].text])
                pair_index.append({"A": indx_a, "B": indx_b})

        chat_input = format_prompts(
            self._prompt, query, batch_content, self._generation_parameters
        )
        generations = await asyncio.gather(
            *[self._io_proc.acreate_chat_completion(c) for c in chat_input]
        )
        win_count = [0] * total_tournament_passages
        for generation, pair in zip(generations, pair_index, strict=True):
            choice = generation.results[0].next_message.content[0].upper()
            if choice == "B":
                win_count[pair["B"]] += 1
                if self._verbose:
                    print("LLM prefers a lower ranked passage")
                    print(f"query: {query}")
                    print(f"passage A: {documents[pair['A']].text}")
                    print(f"passage A position: {pair['A']}")
                    print(f"passage B: {documents[pair['B']].text}")
                    print(f"passage B position: {pair['B']}")
                    print(f"LLM output:{generation.results[0].next_message.content}")
            elif choice == "A":
                win_count[pair["A"]] += 1
            else:
                win_count[pair["A"]] += 0.5
                win_count[pair["B"]] += 0.5
        win_order = sorted(
            range(len(win_count)), key=lambda k: win_count[k], reverse=True
        )
        data_copy = copy.deepcopy(documents[:total_tournament_passages])
        for i, win_indx in enumerate(win_order):
            documents[i] = data_copy[win_indx]

        rerank_results = inputs.model_copy(
            update={"documents": documents[: self._return_top_k]}
        )

        return rerank_results
