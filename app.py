
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
    url_content = st.session_state.metaphor.get_contents([st.session_state.url_ids[query_id+1]])
    summarized = st.session_state.summarizer(url_content.contents[0].extract, min_length=75, max_length=300)
    return summarized

def query_llm(query):
    results = None
    if not st.session_state.metaphor_key or not st.session_state.hugging_api_key or not st.session_state.search_prompt:
        st.warning(f"Please provide the missing fields.")
    else:
        
        if int(query)>=1 and int(query)<=10:
            url_content = url_content_extract(int(query))
            st.session_state.messages.append((list(st.session_state.results_list.keys())[int(query)+1], url_content))

        else:
            st.session_state.metaphor = Metaphor(st.session_state.metaphor_key)
            results = st.session_state.metaphor.search(st.session_state.search_prompt, use_autoprompt=True)
            st.session_state.results_list = defaultdict(str)
            st.session_state.url_ids = []
            for i,r in enumerate(results.results):
                st.session_state.results_list[str(r.title)]=str(r.url)
                st.session_state.url_ids.append(r.id)
            st.session_state.messages.append((query, st.session_state.results_list))
    

def boot():
    #
    input_fields()
    #
    if st.button("Submit prompt"):
        query_llm(-1)
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