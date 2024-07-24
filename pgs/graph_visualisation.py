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
        MATCH (p)<-[:CITES]-(top_paper:Paper)<-[:CITES]-(other_paper:Paper)
        WITH top_paper, COUNT(other_paper) as citation_count
        ORDER BY citation_count DESC
        LIMIT 3
        RETURN top_paper
    }
    RETURN p, top_paper, a
    """.replace("$list", str(paper_ids))
    results = graphDbInstance.query(query)
    return results

def _get_citation_relationships(arxiv_ids: List[str], graphDbInstance: Neo4jGraph):
    query = r"""
    MATCH (p:Paper)
    WHERE p.id in $list
    WITH COLLECT(ELEMENTID(p)) as paper_ids
    CALL apoc.algo.cover(paper_ids)
    YIELD rel
    WITH COLLECT(rel) AS citations
    RETURN [r IN citations | [startNode(r).id, endNode(r).id]] AS node_pairs
    """.replace("$list", str(arxiv_ids))
    results = graphDbInstance.query(query)
    return results[0]["node_pairs"]

def _create_networkx_graph(paper_ids: List[str], graphDbInstance: Neo4jGraph):
    data = _get_raw_auxillary_context_for_papers(paper_ids, graphDbInstance)
    unique_papers = set()
    G = nx.DiGraph()
    for record in data:
        p, top_paper, author = record['p'], record['top_paper'], record['a']
        unique_papers.update([p['id'], top_paper['id']])
        G.add_node(top_paper['id'], label=top_paper['title'], color='salmon')
        G.add_node(p['id'], label=p['title'], color='purple')
        G.add_node(author['name'], label=author['name'], color='yellow')
        G.add_edges_from([
            (p['id'], author['name'], {'label': 'AUTHORED_BY'}),
        ])
    node_pairs = _get_citation_relationships(list(unique_papers), graphDbInstance)
    for pair in node_pairs:
        G.add_edges_from([
            (pair[0], pair[1], {'label': 'CITES'}),
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
