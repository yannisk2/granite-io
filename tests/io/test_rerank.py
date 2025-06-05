# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the retrieval intrinsic.
"""

# Standard
import os

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
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


def test_rerank_request_processor():  # pylint: disable=redefined-outer-name
    temp_data_dir = "data/test_retrieval"
    corpus_name = "govt"
    embeddings_location = f"{temp_data_dir}/{corpus_name}_embeds.parquet"
    if not os.path.exists(embeddings_location):
        download_mtrag_embeddings(
            _EMBEDDING_MODEL_NAME, corpus_name, embeddings_location
        )
    model_name = "ibm-granite/granite-3.3-8b-instruct"
    server = LocalVLLMServer(model_name)
    server.wait_for_startup(200)
    backend = server.make_backend()
    io_proc = make_io_processor(model_name, backend=backend)
    retriever = InMemoryRetriever(embeddings_location, _EMBEDDING_MODEL_NAME)
    request_processor = RetrievalRequestProcessor(retriever, top_k=32)
    rag_chat_input = request_processor.process(_EXAMPLE_CHAT_INPUT)[0]
    rerank_processor = RerankRequestProcessor(
        io_proc, rerank_top_k=16, return_top_k=16, verbose=True
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
    print(match)
    assert not all(match)
