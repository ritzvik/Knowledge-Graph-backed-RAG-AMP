# Knowledge Graph backed RAG AMP

This repository demonstrates how to power a RAG(Retrieval Augmented Generation) application with a knowledge graph(supported by graph DBs like [Neo4j](https://neo4j.com/)) to capture relationships and contexts not easily accessible if vector databases are being used in a RAG pipeline

## AMP Overview

In this AMP, we create a corpus of significant AI/ML papers from [arXiv](https://arxiv.org/), and that first degree citations of these seed papers, resulting in a database of ~350 papers. We expect that these papers would atleast contain some informaton about the latest developments in Artificial Intelligence/Machine Learning/Large Language Models.

Instead of powering the RAG application 