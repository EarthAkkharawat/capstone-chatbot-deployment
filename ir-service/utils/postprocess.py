import logging
import time
import numpy as np
import faiss

from  utils.database import embed_documents

class PostProcess():
    
    def L2_similarity_search_from_dir(response):

        query = response["query"]
        answer = response["result"]
        sources = response["source_documents"]
        vectordb = embed_documents(sources)
        doc_and_score = vectordb.similarity_search_with_score(answer)
        
        l2 = [e[-1] for e in doc_and_score]
        s = [e[0] for e in doc_and_score]

        srcs = ''.join(['> '+ref.metadata["source"]+' :\n\n'+ref.page_content+'\n\n' for ref in s]).strip()
        return {"time": response['time'], "query": query, "answer": answer, "srcs": srcs, "l2_stats": [max(l2), np.mean(l2), min(l2), len(l2)]}
        # f"""Time used: {response['time']:.4f}\nQuestion: {query}\nAnswer: {answer}\n\nSource:\n{srcs}\n\nMax: {max(l2)},\nMean: {np.mean(l2)},\nMin: {min(l2)},\n(from {len(l2)} documents)\n\n"""
        # return f"""Time used: {tf-ti}\nAnswer: {answer}\n\n1.) {doc_and_score[0]}\nL2 Distance Score: {doc_and_score[0][-1]}\n\n2.) {doc_and_score[1]}\nL2 Distance Score: {doc_and_score[1][-1]}\n\nL2 Score Max: {max(l2)}, Mean: {np.mean(l2)}, Min: {min(l2)}, (from {len(l2)} documents)"""

    def get_cosine_similarity(response):
        # TODO: https://github.com/deepset-ai/haystack/issues/1337
        return
