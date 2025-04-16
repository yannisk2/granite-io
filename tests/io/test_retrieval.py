# SPDX-License-Identifier: Apache-2.0

"""
Test cases for the retrieval intrinsic.
"""

# Standard
import os
import pathlib
import tempfile

# Third Party
import pytest
import torch

# Local
from granite_io.io.granite_3_2.input_processors.granite_3_2_input_processor import (
    Granite3Point2Inputs,
)
from granite_io.io.retrieval import (
    InMemoryRetriever,
    RetrievalRequestProcessor,
    compute_embeddings,
    write_embeddings,
)
from granite_io.io.retrieval.util import (
    download_mtrag_corpus,
    download_mtrag_embeddings,
    read_mtrag_corpus,
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

_DATA_DIR = pathlib.Path("data/test_retrieval_temp")
_EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"


@pytest.fixture
def govt_embeddings_file():
    """
    :returns: a pre-indexed copy of the MTRAG benchmark's "govt" data set.
     Uses cached data if available.
    """
    if not os.path.exists(_DATA_DIR):
        os.makedirs(_DATA_DIR)
    target_file = _DATA_DIR / "govt_embeds.parquet"
    if not os.path.exists(target_file):
        download_mtrag_embeddings(_EMBEDDING_MODEL_NAME, "govt", target_file)
    return target_file


@pytest.fixture
def govt_docs_file():
    """
    :returns: a copy of the MTRAG benchmark's "govt" data set.
       Uses cached data if available.
    """
    if not os.path.exists(_DATA_DIR):
        os.makedirs(_DATA_DIR)
    target_file = _DATA_DIR / "govt.jsonl.zip"
    if not os.path.exists(target_file):
        downloaded_file = download_mtrag_corpus(_DATA_DIR, "govt")
        assert downloaded_file == target_file
    return target_file


def test_make_embeddings(govt_docs_file):
    """
    Verify that embedding creation is working by creating embeddings for the first
    10 documents in the "govt" corpus from the MTRAG benchmark.
    """
    full_govt = read_mtrag_corpus(govt_docs_file)
    corpus = full_govt.slice(0, 10)
    embeddings = compute_embeddings(corpus, _EMBEDDING_MODEL_NAME)
    assert embeddings.column("embedding").to_pylist()[0][:10] == pytest.approx(
        [
            -0.11038033664226532,
            0.27693408727645874,
            -0.11863572895526886,
            -0.0792723074555397,
            0.20247098803520203,
            0.09491363912820816,
            0.6091732978820801,
            0.0905776172876358,
            0.10194987803697586,
            -0.011982650496065617,
        ]
    )

    # Round-trip through a file and make sure we get the embeddings back.
    with tempfile.TemporaryDirectory() as tmpdir:
        file_loc = write_embeddings(tmpdir, "test", embeddings)
        retriever = InMemoryRetriever(file_loc, _EMBEDDING_MODEL_NAME)
        # pylint: disable=protected-access
        assert retriever._embeddings[1] == pytest.approx(
            torch.tensor(embeddings.column("embedding").to_pylist()[1])
        )


def test_in_memory_retriever(govt_embeddings_file):
    """Verify basic functionality of the InMemoryRetriever class"""
    retriever = InMemoryRetriever(govt_embeddings_file, _EMBEDDING_MODEL_NAME)
    result = retriever.retrieve(_EXAMPLE_CHAT_INPUT.messages[-1].content)
    assert result.column("id").to_pylist() == [
        "730a96e7b2c4ce22",
        "6738a5f779365684",
        "c7030403177d9dfc",
        "97867ac98aa24313",
        "6738a5f779365684",
    ]


def test_retrieval_request_processor(govt_embeddings_file):
    """Basic test of the RequestProcessor that performs RAG retrieval"""
    retriever = InMemoryRetriever(govt_embeddings_file, _EMBEDDING_MODEL_NAME)
    request_processor = RetrievalRequestProcessor(retriever, top_k=3)
    results = request_processor.process(_EXAMPLE_CHAT_INPUT)
    assert len(results) == 1
    # print(results[0].documents)
    assert [d.doc_id for d in results[0].documents] == [
        "730a96e7b2c4ce22",
        "6738a5f779365684",
        "c7030403177d9dfc",
    ]
