from typing import List, Dict
from langchain.graphs import Neo4jGraph
import networkx as nx
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components

import utils.constants as const
from utils.neo4j_utils import get_neo4j_credentails, is_neo4j_server_up, wait_for_neo4j_server

with st.spinner("Spinning up the Neo4j server..."):
    if not is_neo4j_server_up():
        wait_for_neo4j_server()

    graph = Neo4jGraph(
        username=get_neo4j_credentails()["username"],
        password=get_neo4j_credentails()["password"],
        url=get_neo4j_credentails()["uri"],
    )


def _get_all_papers(graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WITH p, COUNT {(p)<-[:CITES]-(:Paper)} AS citation_count
    ORDER BY citation_count DESC
    RETURN p
    """
    results = graphDbInstance.query(query)
    return results

def _get_all_citation_relationships(graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WITH COLLECT(ELEMENTID(p)) as paper_ids
    CALL apoc.algo.cover(paper_ids)
    YIELD rel
    WITH COLLECT(rel) AS citations
    RETURN [r IN citations | [startNode(r).id, endNode(r).id]] AS node_pairs
    """
    results = graphDbInstance.query(query)
    return results[0]["node_pairs"]

def _create_full_knowledege_base_networkX_graph(graphDbInstance: Neo4jGraph) -> nx.Graph:
    def _get_hover_data(paper: Dict):
        hover_string = paper['title'] + "\n"
        hover_string += "Arxiv ID: " + paper['id'] + "\n"
        hover_string += "Published: " + paper['published'].to_native().strftime("%B %d, %Y")
        return hover_string
    data = _get_raw_papers(graphDbInstance)
    unique_papers = set()
    G = nx.DiGraph()
    for record in data:
        paper = record['p']
        unique_papers.add(paper['id'])
        G.add_node(paper['id'], color='blue', title=_get_hover_data(paper), node_type='Paper')
    node_pairs = _get_all_citation_relationships(graphDbInstance)
    for pair in node_pairs:
        G.add_edges_from([
            (pair[0], pair[1], {'label': 'CITES'}),
        ])
    return G


def visualize_full_graph(graphDbInstance: Neo4jGraph):
    G = _create_full_knowledege_base_networkX_graph(graphDbInstance)
    net = Network(notebook=True)
    net.from_nx(G)
    net.show(const.TEMP_VISUAL_FULL_GRAPH_PATH)

paper_col, viz_col = st.columns([1, 1], gap="small")

all_papers_data = _get_all_papers(graph)
for record in all_papers_data:
    paper = record['p']
    arxiv_id = paper['id']
    arxiv_link = paper['arxiv_link']
    published_string = paper['published'].to_native().strftime("%B %d, %Y")
    paper_title = paper['title']
    paper_col.markdown(f"""
**Arxiv ID**: [{arxiv_id}]({arxiv_link})  
**Title**: {paper_title}  
**Published On**: {published_string}                    
""")
    paper_col.markdown("---")
