from __future__ import annotations
import os
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/excel.html?highlight=xlsx#microsoft-excel
from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)


# https://python.langchain.com/en/latest/_modules/langchain/document_loaders/excel.html#UnstructuredExcelLoader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

# Can be changed to a specific number
INGEST_THREADS = os.cpu_count() or 8


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class == TextLoader:
        loader = TextLoader(file_path, encoding="utf-8")
    elif loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        data_list = [future.result() for future in futures]
        # return data and file paths
        return (data_list, filepaths)


def loadDocuments(
    source_dir: str, chunk_size=1000, chunk_overlap=200
) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    # chunksize = round(len(paths) / n_workers)
    chunksize = max(round(len(paths) / n_workers), 1)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            contents, _ = future.result()
            docs.extend(contents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents: list[Document]
    documents = text_splitter.split_documents(docs)
    # documents = char_data_splitter(docs, chunk_size, chunk_overlap)
    return documents


def char_data_splitter(
    data, chunk_size=2000, chunk_overlap=200, separators=[" ", ",", "\n"]
):
    documents = []
    for datum in data:
        documents.extend(
            split_data_into_documents(datum, chunk_size, chunk_overlap, separators)
        )
    return documents


# Split single data (1 file) into multiple smaller documents of chunk_size
def split_data_into_documents(datum, chunk_size, chunk_overlap, separators):
    if chunk_overlap >= 0.5 * chunk_size:
        raise ValueError("chunk_overlap must be less than half of chunk_size")
    metadata = {"source": datum.metadata["source"]}
    page_content = datum.page_content.strip()
    len_content = len(page_content)
    docs = []
    i = 0

    # Append each smaller document into document lists
    while i < len_content:
        end = find_chunk_end(i, chunk_size, page_content, len_content, separators)
        if end >= len_content:
            i = adjust_chunk_start(end, chunk_size, page_content, separators)
            doc = Document(page_content=page_content[i:end], metadata=metadata)
            docs.append(doc)
            break
        else:
            doc = Document(page_content=page_content[i:end], metadata=metadata)
            docs.append(doc)
            i = adjust_chunk_start(end, chunk_overlap, page_content, separators)

    return docs


# Find start position of next chunk by looking for seperators in overlap chunk
def adjust_chunk_start(end, chunk_overlap, page_content, separators):
    start = max(0, end - chunk_overlap)
    while start < end and page_content[start] not in separators:
        start += 1
    return start if start != end else max(0, end - chunk_overlap)


# Find the end position of the chunk by looking for seperators
def find_chunk_end(start, chunk_size, page_content, len_content, separators):
    end = min(start + chunk_size, len_content)
    if end < len_content:
        while end > start and page_content[end] not in separators:
            end -= 1
    return end if end != start else min(start + chunk_size, len_content)
