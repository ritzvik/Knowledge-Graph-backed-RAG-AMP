from typing import Tuple
import logging
from langchain.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.language_models.llms import BaseLLM
import streamlit as st

import utils.constants as const
from utils.cai_model import getCAIHostedOpenAIModels
from utils.arxiv_utils import linkify_text
from utils.neo4j_utils import get_neo4j_credentails, is_neo4j_server_up, reset_neo4j_server, wait_for_neo4j_server
from utils.hybrid_rag import HybridRAG
from utils.vanilla_rag import VanillaRAG
import pgs.commons as st_commons

embedding = st_commons.get_cached_embedding_model()

st.header("Knowledge Graph based RAG Pipeline")
st.subheader("Ask any AI/ML related question")

if st_commons.StateVariables.IS_REMOTE_LLM.value not in st.session_state:
    st.warning("Please select the LLM model first.", icon=":material/warning:")

with st.spinner("Spinning up the Neo4j server..."):
    if not is_neo4j_server_up():
        reset_neo4j_server()
        wait_for_neo4j_server()

    graph = Neo4jGraph(
        username=get_neo4j_credentails()["username"],
        password=get_neo4j_credentails()["password"],
        url=get_neo4j_credentails()["uri"],
    )

    document_index = Neo4jVector(
        embedding=embedding,
        url=get_neo4j_credentails()["uri"],
        username=get_neo4j_credentails()["username"],
        password=get_neo4j_credentails()["password"],
    )

def load_llm() -> Tuple[BaseLLM, str]:
    if st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value]:
        remote_llm = getCAIHostedOpenAIModels( 
            base_url=st.session_state[st_commons.StateVariables.REMOTE_MODEL_ENDPOINT.value],
            model=st.session_state[st_commons.StateVariables.REMOTE_MODEL_ID.value],
            api_key=st.session_state[st_commons.StateVariables.REMOTE_MODEL_API_KEY.value],
            max_tokens=2048,
            temperature=0.3,
            stop=const.llama3_stop_token,
        )
        return remote_llm, const.llama3_bos_token
    else:
        return st_commons.get_cached_local_model(), const.llama3_bos_token

def generate_responses(input_text):
    col1, col2, col3 = st.columns(3, gap="medium")
    with st.status("Generating Responses...", expanded=True) as status:
        st.write("Loading the LLM model...")
        llm, bos_token = load_llm()
        # since remote model is more powerful.
        if st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value]:
            top_k = 7
        else:
            top_k = 5

        st.write("Generating response from Vanilla RAG...")
        v=VanillaRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        streaming_answer_vanilla = v.stream(input_text)
        logging.info("generated response from Vanilla RAG")
        col1.markdown("## Vanilla RAG")
        col1.write_stream(streaming_answer_vanilla)

        st.write("Generating response from Hybrid RAG...")
        h=HybridRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        answer_hybrid = h.invoke(input_text)
        logging.info("generated response from Hybrid RAG")
        col2.markdown("## Hybrid RAG")
        col2.markdown(linkify_text(answer_hybrid))

        st.write("Generating follow-up details from Hybrid RAG...")
        answer_followup = h.invoke_followup()
        logging.info("generated follow-up answer")
        col3.markdown("## Follow-up details from Hybrid RAG")
        col3.markdown(linkify_text(answer_followup))

        status.update(label="Answer Generation Complete", state="complete", expanded=False)    

with st.form('my_form'):
    text = st.text_area('Enter question:', 'How does knowledge graph help with RAG pipelines?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_responses(text)
