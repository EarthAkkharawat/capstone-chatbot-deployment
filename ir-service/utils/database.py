import logging
import os
import torch

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS

# from enum.embedding_model import EmbeddingModel


def load_embedding_model(embedding_model_name="intfloat/multilingual-e5-small"):

    if torch.cuda.is_available():
        device_type = "cuda"
    # elif torch.backends.mps.is_available():
    #     device_type = "mps"
    else:
        device_type = "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": device_type},
    )
    return embeddings


def embed_database(
    documents,
    persist_directory,
    embedding_model_name="intfloat/multilingual-e5-small",
    vector_store="faiss",
):

    embeddings = load_embedding_model(embedding_model_name)

    # Embedding temp exists
    if os.path.isdir(persist_directory):
        if vector_store == "faiss":
            vectordb = FAISS.load_local(
                persist_directory, embeddings
            )
        elif vector_store == "chroma":
            vectordb = Chroma(
                embedding_function=embeddings, persist_directory=persist_directory
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )

    # Create embeddings if not exists
    else:
        if vector_store == "faiss":
            vectordb = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
            )
            vectordb.save_local(persist_directory)
        elif vector_store == "chroma":
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
            vectordb.persist()
        else:
            raise NotImplementedError(
                f"Embedding Algorithm {vector_store} is not supported/implemented"
            )

    return vectordb


def embed_documents(
    documents,
    embedding_model_name="intfloat/multilingual-e5-small",
    vector_store="faiss",
):

    embeddings = load_embedding_model(embedding_model_name)

    if vector_store == "faiss":
        vectordb = FAISS.from_documents(
            documents=documents,
            embedding=embeddings,
        )
    elif vector_store == "chroma":
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
        )
        vectordb.persist()
    else:
        raise NotImplementedError(
            f"Embedding Algorithm {vector_store} is not supported/implemented"
        )

    return vectordb
