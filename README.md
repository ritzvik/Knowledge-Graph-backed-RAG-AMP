# Knowledge Graph backed RAG AMP

This repository demonstrates how to power a RAG(Retrieval Augmented Generation) application with a knowledge graph(supported by graph DBs like [Neo4j](https://neo4j.com/)) to capture relationships and contexts not easily accessible if vector databases are being used in a RAG pipeline

## AMP Overview

In this AMP, we create a corpus of significant AI/ML papers from [arXiv](https://arxiv.org/), and from first degree citations of these seed papers, resulting in a database of ~350 papers. We expect that these papers would atleast contain some informaton about the latest developments in Artificial Intelligence/Machine Learning/Large Language Models. The AMP user can ask AI/ML related questions, and the RAG pipeline would try to generate answers from the corpus stored in the database.

The AMP is powered by Knowledge Graph(backed by [Neo4j](https://neo4j.com/) Database in our case).

Integration of knowledge graph in this RAG pipeline aims to acheive two things:
 - Retrieve better quality text chunks by employing a hybrid retrieval strategy to pass on the LLM as context. In traditional retrieval methods, a vector embedding id calculated for an user input, and chunks are retrieved from a vector database(based on cosine similarity). We will employ a reranking methodology using information from knowledge graph to "rerank" the chunks retrieved using vector similarity, and pass the top **k** chunks as context.
 - Enhance the answer by giving auxillary information about the papers used to construct the answer, give recommendation about other related papers and top authors to the user.

The AMP is designed to run on and expects [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) LLM as the backend. The AMP can leverage both remotely hosted LLM (Model URL, ID and API key needs to be passed) or locally(in-session) running LLM. In case of "local" LLM, the AMP is designed to run a 4-bit pre-quantized `Meta-Llama-3-8B-Instruct` model(It is pre-quantised as part of AMP steps, no manual intervention required).

## AMP Setup

### Configurable Options

**HF_TOKEN** : The AMP relies on [Huggingface Token](https://huggingface.co/docs/hub/en/security-tokens) to pull [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model from HuggingFace.

*Note*: Please make sure that the account associated with the HuggingFace token has access to the abovementioned model. It does require filling up a form to obtain access.

## AMP Concepts

Let's quickly go through some terminologies we would use throughout the AMP.

### Knowledge Graph

![Sample Image](assets/sample_image.jpg)
<img src="assets/sample_image.jpg" alt="drawing" width="500"/>

### Semantic Search

### Vector Databases

### Retrieval Augmented Generation (RAG)

### Re-Ranking

## AMP Flow

## AMP Requirements
