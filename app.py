
# Importing libraries
import requests
from transformers import pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from collections import defaultdict

import streamlit as st

from metaphor_python import Metaphor

from pathlib import Path
import os, tempfile

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')


st.set_page_config(page_title="Metaphor")
st.title("Talk to articles")

def input_fields():
    #
    with st.sidebar:
        #
        if "metaphor_key" in st.secrets:
            st.session_state.metaphor_key = st.secrets.metaphor_key
        else:
            st.session_state.metaphor_key = st.text_input("Metaphor API key", type="password")
        #
        if "hugging_api_key" in st.secrets:
            st.session_state.hugging_api_key = st.secrets.hugging_api_key
        else: 
            st.session_state.hugging_api_key = st.text_input("Hugging Face API key", type="password")
        #
    #
    st.session_state.search_prompt = st.text_input(label="Prompt for articles")
    #

def url_content_extract(query_id):
    st.session_state.summarizer = pipeline("summarization")
    embedding = HuggingFaceEmbeddings()
    st.session_state.url_content = st.session_state.metaphor.get_contents([st.session_state.url_ids[query_id-1]])
    summarized = st.session_state.summarizer(st.session_state.url_content.contents[0].extract, min_length=75, max_length=300)
    return summarized

def query_search(query):
    st.session_state.metaphor = Metaphor(st.session_state.metaphor_key)
    results = st.session_state.metaphor.search(st.session_state.search_prompt, use_autoprompt=True)
    st.session_state.results_list = defaultdict(str)
    st.session_state.url_ids = []
    for i,r in enumerate(results.results):
        st.session_state.results_list[str(r.title)]=str(r.url)
        st.session_state.url_ids.append(r.id)
    st.session_state.messages.append((st.session_state.search_prompt, st.session_state.results_list))
    return st.session_state.results_list

def query_llm(query):
    results = None
    if not st.session_state.metaphor_key or not st.session_state.hugging_api_key or not st.session_state.search_prompt:
        st.warning(f"Please provide the missing fields.")
    else: 
        try:
            if int(query)>=1 and int(query)<=10:
                url_content = url_content_extract(int(query))
                st.session_state.messages.append((int(query),str(list(st.session_state.results_list.keys())[int(query)-1])+":"+str(url_content[0]["summary_text"])))
                results = str(list(st.session_state.results_list.keys())[int(query)-1])+":"+str(url_content[0]["summary_text"])
        except:
            embedding = HuggingFaceEmbeddings()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=0, separators=[" ", ",", "\n"])
            texts = text_splitter.split_text(st.session_state.url_content.contents[0].extract)
            db = FAISS.from_texts(texts, embedding)
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.session_state.hugging_api_key
            if "llm" not in st.session_state:
                st.session_state.llm=HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature":0.2, "max_length":256}
                )
            chain = load_qa_chain(st.session_state.llm, chain_type="stuff")
            docs = db.similarity_search(query) 
            output = chain.run(input_documents=docs, question=query)
            st.session_state.messages.append((st.session_state.search_prompt, output))
    return results
    

def boot():
    #
    input_fields()
    #
    if st.button("Submit prompt"):
        query_search(-1)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()