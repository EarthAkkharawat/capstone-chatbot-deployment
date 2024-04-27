import os
import sys
import time
import re
import torch
from dotenv import load_dotenv
from tqdm import tqdm
import json
import numpy as np
import pandas as pd

from pythainlp import word_tokenize, pos_tag
from pythainlp.corpus.common import thai_stopwords
import nltk
from nltk.corpus import stopwords

import logging
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import cohere

from utils.data import loadDocuments
from utils.database import embed_database

from fastapi import FastAPI, APIRouter

load_dotenv()

co = None

root_dir = "."
sys.path.append(root_dir)  # if import module in this project error
if os.name != "nt":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
# %% [markdown]
# ### **setup var**

# %%
# chunk_size = 2000
# chunk_overlap = 200
# embedding_algorithm = "faiss"
# source_directory = f"{root_dir}/ir-service/docs"
# persist_directory = f"{root_dir}/ir-service/tmp/embeddings/{embedding_algorithm}"
# print(root_dir)
# print(persist_directory)

# Original mapper dictionary
mapper = {
    "law_doc-84-89.txt": "761/2566",
    "law_doc-44-46.txt": "1301/2566",
    "law_doc-54-57.txt": "1225/2566",
    "law_doc-12-13.txt": "2525/2566",
    "law_doc-40-43.txt": "1305/2566",
    "law_doc-14-15.txt": "2085/2566",
    "law_doc-64-69.txt": "1090/2566",
    "law_doc-1-5.txt": "2610/2566",
    "law_doc-78-81.txt": "882/2566",
    "law_doc-82-83.txt": "835/2566",
    "law_doc-35-39.txt": "1306/2566",
    "law_doc-16-20.txt": "1574/2566",
    "law_doc-32-34.txt": "1373/2566",
    "law_doc-74-77.txt": "934/2566",
    "law_doc-6-11.txt": "2609/2566",
    "law_doc-90-92.txt": "756/2566",
    "law_doc-47-53.txt": "1300/2566",
    "law_doc-58-63.txt": "1101/2566",
    "law_doc-70-73.txt": "1003/2566",
    "law_doc-21-31.txt": "1542/2566",
}

# Reverse the mapping and format it
mapper_reverse = {f"คดี {case}": filename for filename, case in mapper.items()}

endl = "\n"
# print(root_dir)
exclude_pattern = re.compile(r"[^ก-๙]+")  # |[^0-9a-zA-Z]+


def is_exclude(text):
    return bool(exclude_pattern.search(text))


key_tags = ["NCMN", "NCNM", "NPRP", "NONM", "NLBL", "NTTL"]

thaistopwords = list(thai_stopwords())
nltk.download("stopwords")


def remove_stopwords(text):
    res = [
        word.lower()
        for word in text
        if (word not in thaistopwords and word not in stopwords.words())
    ]
    return res


def keyword_search(question):
    tokens = word_tokenize(question, engine="newmm", keep_whitespace=False)
    pos_tags = pos_tag(tokens)
    noun_pos_tags = []
    for e in pos_tags:
        if e[1] in key_tags:
            noun_pos_tags.append(e[0])
    noun_pos_tags = remove_stopwords(noun_pos_tags)
    noun_pos_tags = list(set(noun_pos_tags))
    return noun_pos_tags


# %%
def find_case_number(text):
    pattern = re.compile(r"(?<!\d)(\d{1,5}/\d{4})(?!\d)")
    match = re.findall(pattern, text)
    if pattern.search(text) and all(e in mapper.values() for e in match):
        return [True, match]
    else:
        return [False, ""]


def keyword_matcher(doc, keywords):
    matched_keywords = []
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword))
        if pattern.search(doc.page_content):
            matched_keywords.append(keyword)
    return matched_keywords


def filter_docs_by_keywords(docs, keywords, question):
    filtered_docs = []
    matches = []
    for doc in docs:
        matched_keywords = []
        if find_case_number(question)[0]:
            case_num = find_case_number(question)[1]
            for num in case_num:
                pattern = re.compile(re.escape(num))
                if pattern.search(doc.page_content):
                    matched_keywords = keyword_matcher(doc, keywords)
                    if len(matched_keywords) >= min(3, len(keywords)):
                        matches.append(matched_keywords)
                        filtered_docs.append(doc)
            continue
        matched_keywords = keyword_matcher(doc, keywords)
        if len(matched_keywords) >= min(2, len(keywords)):
            matches.append(matched_keywords)
            filtered_docs.append(doc)
    return filtered_docs, matches


# %%
def parse_source_docs(source_docs):
    if source_docs is not None:
        results = []
        for res in source_docs:
            if res.metadata["source"].split("/")[-1] in mapper:
                context = f"""คดีหมายเลข {mapper[res.metadata["source"].split("/")[-1]]}\n{res.page_content}"""
                results.append(context)
            else:
                results.append(res.page_content)
        # srcs = [f"""<<<{res.metadata["source"].split("/")[-1]}>>>\n<<<case #{mapper[res.metadata["source"].split("/")[-1]]}>>>\n{res.page_content}""" for res in source_docs]
        result = "\n\n".join(results)
        return result
    else:
        return []


def parse_matched_keywords(matched_keywords):
    if matched_keywords is not None:
        result = "\n".join(str(keyword) + "," for keyword in matched_keywords)
    else:
        result = []
    return result


def retriever(question, documents, vector_database):
    global co
    try:
        if question in ["", "-", None]:
            raise Exception("No question")
        ti = time.time()

        # keywords search
        keywords = keyword_search(question)
        keywords_filtered_docs, matched_keywords = filter_docs_by_keywords(
            documents, keywords, question
        )
        if len(keywords_filtered_docs) == 0:
            return {
                "time": 0,
                "question": question,
                "reranked_docs": "",
            }

        # context search
        retrieved_docs = []
        if not find_case_number(question)[0]:
            retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={"score_threshold": 0.6})
            retrieved_docs = retriever.get_relevant_documents(question)
        if len(retrieved_docs) == 0:
            return {
                "time": tf - ti,
                "question": question,
                "reranked_docs": "",
            }
            
        # rerank
        relevant_src_docs = keywords_filtered_docs + retrieved_docs
        if len(relevant_src_docs) == 0:
            return {
                "time": tf - ti,
                "question": question,
                "reranked_docs": "",
            }
        relevant_docs = [doc.page_content for doc in relevant_src_docs]
        if co is None:
            co = cohere.Client(os.getenv("COHERE"))
        rerank_hits = co.rerank(
            query=question,
            documents=relevant_docs,
            model="rerank-multilingual-v2.0",
            top_n=1,
        )
        results = [relevant_src_docs[hit.index] for hit in rerank_hits.results]
        parse_reranked_docs = parse_source_docs(results)
        tf = time.time()
        
        del keywords, keywords_filtered_docs, matched_keywords, retrieved_docs, relevant_src_docs, relevant_docs, rerank_hits, results
        
        # return f"""> Time: {tf-ti}\n\n> Question: {question}\n\n> Answer: {result}\n\n> Source docs:\n{relevant_source_docs}"""
        return {
            "time": tf - ti,
            "question": question,
            "reranked_docs": parse_reranked_docs,
        }
    except Exception as e:
        print(f"{question} @{e}")
        return {
            "error": str(e),
            "source_doc": [],
            "response": "",
            "time": "",
            "source": "",
        }

contents = None
documents_specific = None
documents_general = None
documents = None
vectordb = None

def main(question):
    global contents
    global documents_specific
    global documents_general
    global documents
    global vectordb
    
    if contents is None:
        with open(f"{root_dir}/specific_case_knowledge.txt", "r", encoding="utf-8") as f:
            content = f.read()
        contents = content.split("\n\n")
        
    if documents_specific is None:    
        documents_specific = [
            Document(
                page_content=f"{endl.join(c.split(endl)[1:])}",
                metadata={
                    "source": f"docs/{mapper_reverse[c.split(endl)[0]]}",
                    "category": "specific",
                },
            )
            for c in contents
        ]
    if documents_general is None:    
        documents_general = loadDocuments(
            source_dir=f"{root_dir}/_ตรวจแล้ว", chunk_size=10e12, chunk_overlap=0
        )
        for i in range(len(documents_general)):
            documents_general[i].metadata["source"] = (
                documents_general[i].metadata["source"].replace(f"{root_dir}/", "")
            )
            documents_general[i].metadata["category"] = "general"
    if documents is None:
        documents = documents_general + documents_specific

    persist_directory = f"{root_dir}/vectordb"
    if vectordb is None:
        vectordb = embed_database(documents=documents, persist_directory=persist_directory)
    # question = "ขอดูอย่างคดีที่มีการพิพากษาของศาลฎีกาต่างจากศาลอุทธรณ์หน่อยได้ไหมครับ"
    qa_res = retriever(question, documents, vectordb)
    print(qa_res)
    torch.cuda.empty_cache()
    return qa_res