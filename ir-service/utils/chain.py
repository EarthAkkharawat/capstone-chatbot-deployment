import logging
import time

from langchain.chains import RetrievalQA

# from utils.postprocess import PostProcess
# from utils.integrations.langfuse import get_langfuse_callback_handler

class Chain():

    def __init__(self, llm, retriever, prompt, return_source, memory=None):
        self.return_source = return_source
        chain_kwargs = {"prompt": prompt, "memory": memory} if memory is not None else {"prompt": prompt}
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs=chain_kwargs,
            return_source_documents=True
        )

    def run(self, query, langfuse=False):
        ti = time.time()
        # response = self.chain(query, callbacks=[get_langfuse_callback_handler()] if langfuse else None)
        response = self.chain(query, callbacks=None)
        # response = self.chain({"query": query})
        tf = time.time()
        response["return_source"] = self.return_source
        response["time"] = tf - ti
        return response
    
    def parse_response(self, response):
        if response["return_source"]:
            source = "\n\n----------------------------------SOURCE DOCUMENTS---------------------------"
            for ref in response["source_documents"]:
                source += f'\n\n> {ref.metadata["source"]} :\n\n {ref.page_content}'
        return f'\n\n> Question: {response["query"]}\n\n> Answer: {response["result"]}\n\n> Time: {response["time"]} seconds{source if response["return_source"] else ""}\n\n'

    def parse_response2(self, response):
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
            "law_doc-21-31.txt": "1542/2566"
        }
        fresult = {"response": response["result"], "time": response["time"]}
        if response["return_source"]:
            if "source_documents" not in response:
                fresult["source_doc"] = []
            else:
                srcs = [f"""<<<{res.metadata["source"].split("/")[-1]}>>>\n<<<case #{mapper[res.metadata["source"].split("/")[-1]]}>>>\n{res.page_content}""" for res in response["source_documents"]]
                fresult["source"] = "\n\n".join(srcs)
                fresult["source_doc"] = response["source_documents"]
            if fresult["source_doc"] in ["",None]:
                fresult["source_doc"] = []
        else:
            fresult["source_doc"] = []
        return fresult