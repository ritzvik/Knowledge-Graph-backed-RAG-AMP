from typing import List
from langchain.graphs import Neo4jGraph
import networkx as nx
from pyvis.network import Network
import streamlit as st

import utils.constants as const

def _get_raw_auxillary_context_for_papers(paper_ids: List[str], graphDbInstance: Neo4jGraph) -> str:
    query = r"""
    MATCH (p:Paper)
    WHERE p.id in $list
    CALL {
        WITH p
        MATCH (p)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(other_paper:Paper)
        WITH a, COUNT(other_paper) as other_paper_count
        ORDER BY other_paper_count DESC
        LIMIT 3
        RETURN a
    }
    CALL {
        WITH p
        MATCH (p)<-[:CITES]-(cited_by:Paper)<-[:CITES]-(other_paper:Paper)
        WITH cited_by, COUNT(other_paper) as citation_count
        ORDER BY citation_count DESC
        LIMIT 3
        RETURN cited_by
    }
    RETURN p, cited_by, a
    """.replace("$list", str(paper_ids))
    results = graphDbInstance.query(query)
    return results

def _create_networkx_graph(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    data = _get_raw_auxillary_context_for_papers(paper_ids, graphDbInstance)
    G = nx.DiGraph()
    for record in data:
        p, cited_by, author = record['p'], record['cited_by'], record['a']
        G.add_node(cited_by['id'], label=cited_by['title'], color='salmon')
        G.add_node(p['id'], label=p['title'], color='purple')
        G.add_node(author['name'], label=author['name'], color='yellow')
        G.add_edges_from([
            (cited_by['id'], p['id'], {'label': 'CITES'}),
            (p['id'], author['name'], {'label': 'AUTHORED_BY'}),
        ])
    return G

def visualize_graph(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    progress_bar = st.progress(20, "Running cypher query for auxiliary context.")
    G = _create_networkx_graph(paper_ids, graphDbInstance)
    progress_bar.progress(60, "Rendering graph.")
    net = Network(notebook=True)
    net.from_nx(G)
    progress_bar.progress(90, "Saving graph.")
    net.show(const.TEMP_VISUAL_GRAPH_PATH)
    progress_bar.empty()
