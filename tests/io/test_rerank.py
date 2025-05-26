# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the retrieval intrinsic.
"""

# Standard
import pathlib

# Third Party
import pytest

# Local
from granite_io import make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
)
from granite_io.io.llmrerank import RerankRequestProcessor
from granite_io.io.retrieval import (
    InMemoryRetriever,
    RetrievalRequestProcessor,
)

_EXAMPLE_CHAT_INPUT = Granite3Point2Inputs.model_validate(
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

_DATA_DIR = pathlib.Path("data/test_retrieval")
_EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


@pytest.fixture
def govt_embeddings_file():
    """
    :returns: a pre-indexed copy of a tiny slice of the MTRAG benchmark's "govt" data
     set.
    """
    target_file = _DATA_DIR / "govt10_embeds.parquet"
    return target_file


@pytest.fixture
def govt_docs_file():
    """
    :returns: a copy of a tiny slice of the MTRAG benchmark's "govt" data set.
    """
    target_file = _DATA_DIR / "govt10.jsonl.zip"
    return target_file


def test_rerank_request_processor(govt_embeddings_file):  # pylint: disable=redefined-outer-name
    """Basic test of the RequestProcessor that performs RAG retrieval"""
    model_name = "ibm-granite/granite-3.2-8b-instruct"
    server = LocalVLLMServer(model_name)
    server.wait_for_startup(200)
    backend = server.make_backend()
    io_proc = make_io_processor(model_name, backend=backend)
    retriever = InMemoryRetriever(govt_embeddings_file, _EMBEDDING_MODEL_NAME)
    request_processor = RetrievalRequestProcessor(retriever, top_k=32)
    rag_chat_input = request_processor.process(_EXAMPLE_CHAT_INPUT)[0]
    rerank_processor = RerankRequestProcessor(io_proc, rerank_top_k=16, return_top_k=16)
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
