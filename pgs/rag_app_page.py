from typing import Tuple, List
import logging
from langchain.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_core.language_models.llms import BaseLLM
import streamlit as st
import streamlit.components.v1 as components

import utils.constants as const
from utils.cai_model import getCAIHostedOpenAIModels
from utils.arxiv_utils import linkify_text, PaperChunk, IngestablePaper
from utils.neo4j_utils import get_neo4j_credentails, is_neo4j_server_up, wait_for_neo4j_server
from utils.knowledge_graph_rag import KnowledgeGraphRAG
from utils.vanilla_rag import VanillaRAG
import pgs.commons as st_commons
import pgs.graph_visualisation as st_graph_viz

embedding = st_commons.get_cached_embedding_model()

st.header("Knowledge Graph powered RAG")
st.subheader("Ask any AI/ML related question")

if st_commons.StateVariables.IS_REMOTE_LLM.value not in st.session_state:
    # Default to local LLM in case of no selection.
    st_commons.get_cached_local_model()
    st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value] = False

with st.spinner("Spinning up the Neo4j server..."):
    if not is_neo4j_server_up():
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

def format_context(paper_chunks: List[PaperChunk]):
    chunk_mappings = dict()
    for chunk in paper_chunks:
        if chunk.paper.arxiv_id not in chunk_mappings:
            chunk_mappings[chunk.paper.arxiv_id] = {'paper': chunk.paper, 'chunks': []}
        chunk_mappings[chunk.paper.arxiv_id]['chunks'].append(chunk.text)
    for i, val in enumerate(chunk_mappings.values()):
        paper: IngestablePaper = val['paper']
        st.markdown(f"""
### Paper {i+1}
**Arxiv ID**: [{paper.arxiv_id}]({paper.arxiv_link})  
**Title**: {paper.title}  
**Citiation Count**: {paper.citation_count}  
*Chunks Used*:
""")
        with st.container(border=True, height=100):
            st.markdown("\n\n".join(val['chunks']))

def generate_responses_v2(input_text):
    status_container = st.container()
    kg_col, vanilla_col = st.columns([0.65, 0.35], gap="small")
    st.markdown("""
        <style>
            [data-testid="column"]:nth-child(2){
                background-color: #F6F6F6;
            }
            [data-testid="column"]:nth-child(1){
                background-color: #FAFAFA;
            }
        </style>
        """, unsafe_allow_html=True
    )
    kg_col_header, vanilla_col_header = kg_col.container(border=False), vanilla_col.container(border=False)

    kg_col_header.markdown("<h2 style='text-align: center;'>Knowledge Graph RAG </h2>", unsafe_allow_html=True)
    vanilla_col_header.markdown("<h2 style='text-align: center;'>Vanilla RAG </h2>", unsafe_allow_html=True)

    kg_answer_container = kg_col.container(height=250, border=False)
    vanilla_answer_container = vanilla_col.container(height=250, border=False)
    
    with status_container.status("Generating Responses...", expanded=True) as status:
        status.write("Loading the LLM model...")
        llm, bos_token = load_llm()
        # since remote model is more powerful.
        if st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value]:
            top_k = 7
        else:
            top_k = 5

        status.write("Generating response from Knowledge Graph RAG...")
        k=KnowledgeGraphRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        answer_kg = k.invoke(input_text)
        papers_used_in_kg_answer = k.used_papers
        kg_answer_container.markdown(linkify_text(answer_kg))

        kg_chunks_used = k.retrieve_chunks(input_text)
        kg_context_expander = kg_col.expander("Context from Knowledge Graph enhanced vector search", expanded=False)
        with kg_context_expander:
            format_context(kg_chunks_used)

        status.write("Generating additional details about the answer...")
        kg_additional_container = kg_col.container(height=250, border=False)
        kg_additional_context = k.invoke_followup()
        kg_additional_container.markdown("### Related Papers and Authors(from the knowledge graph)")
        kg_additional_container.markdown(linkify_text(kg_additional_context))
        kg_col.markdown("---")

        st_graph_viz.visualize_graph(papers_used_in_kg_answer, graph)
        htmlfile = open(const.TEMP_VISUAL_GRAPH_PATH, 'r', encoding='utf-8')
        htmlfile_source_code = htmlfile.read()
        with kg_col:
            components.html(htmlfile_source_code, height=350, scrolling=True)
        kg_col.markdown(st_commons.graph_visualisation_markdown)

        status.write("Generating response from Vanilla RAG...")
        v=VanillaRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        answer_vanilla = v.invoke(input_text)
        vanilla_answer_container.markdown(linkify_text(answer_vanilla))

        vanilla_chunks_used = v.retrieve_chunks(input_text)
        vanilla_context_expander = vanilla_col.expander("Context from vector search", expanded=False)
        with vanilla_context_expander:
            format_context(vanilla_chunks_used)

        status.update(label="Answer Generation Complete", state="complete", expanded=False)


def generate_responses(input_text):
    status_container = st.container()
    col1, col2 = st.columns([0.4, 0.6], gap="small")
    col1_header = col1.container(border=False)
    col2_header = col2.container(border=False)
    vanilla_container = col1.container(height=st_commons.response_container_height, border=True)
    hybrid_container = col2.container(height=st_commons.response_container_height, border=True)
    hybrid_response = hybrid_container.container(height=int(st_commons.response_container_height*0.60), border=False)
    hybrid_container.markdown("---")
    hybrid_folllow_up = hybrid_container.container(height=int(st_commons.response_container_height*0.30), border=False)
    with status_container.status("Generating Responses...", expanded=True) as status:
        status.write("Loading the LLM model...")
        llm, bos_token = load_llm()
        # since remote model is more powerful.
        if st.session_state[st_commons.StateVariables.IS_REMOTE_LLM.value]:
            top_k = 7
        else:
            top_k = 5

        status.write("Generating response from Vanilla RAG...")
        v=VanillaRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        answer_vanilla = v.invoke(input_text)
        logging.info("generated response from Vanilla RAG")
        col1_header.markdown("## Vanilla RAG")
        vanilla_container.markdown(linkify_text(answer_vanilla))

        status.write("Generating response from Hybrid RAG...")
        h=KnowledgeGraphRAG(graphDbInstance=graph, document_index=document_index, llm=llm, top_k=top_k, bos_token=bos_token)
        answer_hybrid = h.invoke(input_text)
        papers_used_in_hybrid = h.used_papers
        logging.info("generated response from Hybrid RAG")
        col2_header.markdown("## Hybrid RAG")
        hybrid_response.markdown(linkify_text(answer_hybrid))

        status.write("Generating follow-up details from Hybrid RAG...")
        answer_followup = h.invoke_followup()
        logging.info("generated follow-up answer")
        hybrid_folllow_up.markdown("### Follow-up details")
        hybrid_folllow_up.markdown(linkify_text(answer_followup))

        status.update(label="Answer Generation Complete", state="complete", expanded=False)
    
    st.markdown("""---""")
    st.markdown(st_commons.graph_visualisation_markdown)
    st_graph_viz.visualize_graph(papers_used_in_hybrid, graph)
    htmlfile = open(const.TEMP_VISUAL_GRAPH_PATH, 'r', encoding='utf-8')
    htmlfile_source_code = htmlfile.read()
    components.html(htmlfile_source_code, height=800, scrolling=True)

with st.form('my_form'):
    input_text = ""
    st.session_state[st_commons.StateVariables.QUESTION_FROM_DROPDOWN.value] = st.selectbox(
        'Select from a list of sample questions',
        st_commons.example_questions,
        index=None,
        placeholder="Select an example question...",
    )
    input_text = st.text_area(
        '...Or ask your own AI/ML related question here.',
        value="",
        disabled=(
            (st_commons.StateVariables.QUESTION_FROM_DROPDOWN.value in st.session_state)
            and
            (st.session_state[st_commons.StateVariables.QUESTION_FROM_DROPDOWN.value] is not None)
        ),
        height=10,
    )
    submitted = st.form_submit_button('Submit')
    question_from_dropdown = st.session_state[st_commons.StateVariables.QUESTION_FROM_DROPDOWN.value]
    if submitted:
        generate_responses_v2(question_from_dropdown if question_from_dropdown is not None else input_text)
