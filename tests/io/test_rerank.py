# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the llm rerank intrinsic.
"""

# Standard
import os

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend import Backend
from granite_io.io.consts import (
    _GRANITE_3_3_MODEL_NAME,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.llmrerank import RerankRequestProcessor
from granite_io.io.retrieval import (
    InMemoryRetriever,
    RetrievalRequestProcessor,
)
from granite_io.io.retrieval.util import download_mtrag_embeddings

_EXAMPLE_CHAT_INPUT = Granite3Point3Inputs.model_validate(
    {
        "messages": [
            {
                "role": "assistant",
                "content": "Welcome to the California Appellate Courts help desk.",
            },
            {
                "role": "user",
                "content": "I need to do some legal research to be prepared for my "
                "oral argument. Can I visit the law library?",
            },
        ],
        "generate_inputs": {
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    }
)

_EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


@pytest.mark.vcr(record_mode="new_episodes")
def test_rerank_request_processor(backend_3_3: Backend):  # pylint: disable=redefined-outer-name
    temp_data_dir = "data/test_retrieval"
    corpus_name = "govt10"
    embeddings_location = f"{temp_data_dir}/{corpus_name}_embeds.parquet"
    if not os.path.exists(embeddings_location):
        download_mtrag_embeddings(
            _EMBEDDING_MODEL_NAME, corpus_name, embeddings_location
        )
    io_proc = make_io_processor(_GRANITE_3_3_MODEL_NAME, backend=backend_3_3)
    retriever = InMemoryRetriever(embeddings_location, _EMBEDDING_MODEL_NAME)
    request_processor = RetrievalRequestProcessor(retriever, top_k=128)
    rag_chat_input = request_processor.process(_EXAMPLE_CHAT_INPUT)[0]
    rerank_processor = RerankRequestProcessor(
        io_proc, rerank_top_k=128, return_top_k=128, verbose=True
    )
    rerank_chat_input = rerank_processor.process(rag_chat_input)
    rerank_doc_ids = []
    retrieval_doc_ids = []
    for doc in rerank_chat_input.documents:
        rerank_doc_ids.append(doc.doc_id)
    for doc in rag_chat_input.documents:
        retrieval_doc_ids.append(doc.doc_id)
    match = [
        id1 == id2 for id1, id2 in zip(rerank_doc_ids, retrieval_doc_ids, strict=False)
    ]
    assert not all(match)
