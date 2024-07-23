import gc
from enum import Enum

import time
import streamlit as st
import torch
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import BaseLLM

import utils.constants as const
from utils.huggingface_utils import load_local_model

@st.cache_resource(show_spinner=False)
def get_cached_local_model() -> BaseLLM:
    progress_bar = st.progress(0, "Running garbage collection.")
    gc.collect()
    progress_bar.progress(30, "Emptying CUDA cache.")
    with torch.no_grad():
        torch.cuda.empty_cache()
    progress_bar.progress(60, f"Loading local {const.local_model_to_be_quantised} model.")
    local_llm = load_local_model()
    progress_bar.progress(99, "Model loaded successfully.")
    time.sleep(1.0)
    progress_bar.empty()
    return local_llm

@st.cache_resource(show_spinner=False)
def get_cached_embedding_model():
    progress_bar = st.progress(20, f"Loading {const.embed_model_name} embedding model.")
    embedding = HuggingFaceEmbeddings(model_name=const.embed_model_name, cache_folder=const.EMBED_PATH)
    progress_bar.progress(100, "Model loaded successfully.")
    progress_bar.empty()
    return embedding

class StateVariables(Enum):
    REMOTE_MODEL_ENDPOINT = "remote_model_endpoint"
    REMOTE_MODEL_ID = "remote_model_id"
    REMOTE_MODEL_API_KEY = "remote_model_api_key"
    IS_REMOTE_LLM = "is_remote_llm"

example_questions = [
    "What is the difference between GPT-3 and GPT-4?",
    "How does LLMs help in achieving artificial general intelligence?",
    "What are agentic workflows in AI?",
    "Where does vector embeddings fit in the RAG pipeline?",
    "How does knowledge graph help in improving the quality of a RAG pipeline?",
]
