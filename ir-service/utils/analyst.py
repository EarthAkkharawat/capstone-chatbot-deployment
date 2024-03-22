import os
import time

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain

# from utils.integrations.langfuse import get_langfuse_callback_handler

def test_llm(llm, question, document):

    docs = [Document(page_content=document.strip(), metadata={"source": "Human"})]
    chain = load_qa_chain(
        llm, 
        chain_type="stuff",
        # prompt=TestPrompt(),
    )
    ti = time.time()

    # reponse format : {'input_documents': [Document(page_content='', metadata={'source': ''})], 'question': '', 'output_text': ''}
    response = chain({"input_documents": docs, "question": question})

    tf= time.time()

    # add metadata / change key to source_documents / result / query
    response["time"] = tf - ti
    response["source_documents"] = response.pop("input_documents")
    response['result'] = response.pop('output_text')
    response['query'] = response.pop('question')

    return response

def TestPrompt():
    # template = """จงใช้ข้อมูลที่ให้เพื่อตอบคำถามที่อยู่ส่วนท้ายที่สุดหากตอบไม่ได้ให้บอกว่าไม่ทราบ
    template = """จงใช้ข้อมูลที่ให้เท่านั้นเพื่อตอบคำถามที่อยู่ส่วนท้ายที่สุด หากตอบไม่ได้ให้บอกว่าไม่ทราบ เเละห้ามใช้ความรู้ของตัวเอง

    {context}

    คำถาม: {question}
    คำตอบ:"""
    return PromptTemplate(input_variables=["question","context"], template=template)