# Knowledge Graph backed RAG AMP

This repository demonstrates how to power a RAG(Retrieval Augmented Generation) application with a knowledge graph(supported by graph DBs like [Neo4j](https://neo4j.com/)) to capture relationships and contexts not easily accessible if vector databases are being used in a RAG pipeline

## AMP Overview

In this AMP, we create a corpus of significant AI/ML papers from [arXiv](https://arxiv.org/), and from first degree citations of these seed papers, resulting in a database of ~350 papers. We expect that these papers would atleast contain some informaton about the latest developments in Artificial Intelligence/Machine Learning/Large Language Models. The AMP user can ask AI/ML related questions, and the RAG pipeline would try to generate answers from the corpus stored in the database.

The AMP is powered by Knowledge Graph(backed by [Neo4j](https://neo4j.com/) Database in our case).

Integration of knowledge graph in this RAG pipeline aims to acheive two things:
 - Retrieve better quality text chunks by employing a hybrid retrieval strategy to pass on the LLM as context. In traditional retrieval methods, a vector embedding id calculated for an user input, and chunks are retrieved from a vector database(based on cosine similarity). We will employ a reranking methodology using information from knowledge graph to "rerank" the chunks retrieved using vector similarity, and pass the top **k** chunks as context.
 - Enhance the answer by giving auxillary information about the papers used to construct the answer, give recommendation about other related papers and top authors to the user.

The AMP is designed to run on and expects [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) LLM as the backend. The AMP can leverage both remotely hosted LLM (Model URL, ID and API key needs to be passed) or locally(in-session) running LLM. In case of "local" LLM, the AMP is designed to run a 4-bit pre-quantized `Meta-Llama-3.1-8B-Instruct` model(It is pre-quantised as part of AMP steps, no manual intervention required).

## AMP Setup

### Configurable Options

**HF_TOKEN** : The AMP relies on [Huggingface Token](https://huggingface.co/docs/hub/en/security-tokens) to pull [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model from HuggingFace.

*Note*: Please make sure that the account associated with the HuggingFace token has access to the abovementioned model. It does require filling up a form to obtain access.

## AMP Concepts

Let's quickly go through some terminologies we would use throughout the AMP.

### Knowledge Graph

Knowledge graphs (KGs) organise data from multiple sources, capture information about entities of interest in a given domain or task (like people, places or events), and forge connections between them. In data science and AI, knowledge graphs are commonly used to:

 - Facilitate access to and integration of data sources.
 - Add context and depth to other, more data-driven AI techniques such as machine learning
 - Serve as bridges between humans and systems, such as generating human-readable explanations, or, on a bigger scale, enabling intelligent systems for scientists and engineers. 

![Example Knowledge Graph](./assets/example_knowledge_graph.png)
<span class="caption">An example KG from the AMP showing a paper, the categories the paper belongs to, its authors and other citing papers.</span>

### Semantic Search

Semantic search is a search engine technology that interprets the meaning of words and phrases. The results of a semantic search will return content matching the meaning of a query, as opposed to content that literally matches words in the query.

**Vector Search** is a method for doing semantic search, used heavily in RAG pipelines. Vector search in a RAG context follows a series of steps:
 - We already have a corpus of text and their corresponding [vector embeddings](https://www.elastic.co/what-is/vector-embedding) stored in the database.
 - The vector embedding of the user query is calculated.
 - A vector similarity search is performed accross the database using a calculated embedding as the referance and top *k* results are retreived.

![Vector Search Diagram](./assets/vector-search-diagram.png)
<span class="caption">A simple diagram showing the flow of a vector search. Credits: [Elasticsearch B.V.](https://www.elastic.co/what-is/vector-embedding)</span>

### Vector Databases

Vector databases store the vector embedding of a content(chunk of text, sound or image) along with some corresponding metedata for the content.

 - Vector databases offer well-known and easy-to-use features for data storage, like inserting, deleting, and updating data.
 - Vector databases have specialized capability to perform vector similarity search to retreive vector embeddings stored in the database along with the associated content based off some well known similarity metrics(Cosine similarity, or KNN).

*Note*: In our AMP, the Graph Database (Neo4j) also acts as vector database and have vector similarity search capabilities, thanks to [Graph Data Science(GDS)](https://github.com/neo4j/graph-data-science) plugin.

<img src="./assets/paper_with_chunks.png"  width="50%" height="50%" /><img src="./assets/chunk_attributes.png"  width="40%" height="40%" />

<span class="caption">The diagrams show how chunks are connected to their source paper in the knowledge graph. Each chunk holds the text and the precomputed embedding of the text.</span>

### Retrieval Augmented Generation (RAG)

Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.

Steps involvded in a RAG pipeline (Ref:[Langchain](https://blog.langchain.dev/tutorial-chatgpt-over-your-data/)):
 - Ingestion of Data
   - Load data sources into text
   - Chunk the text: This is necessary because language models generally have a limit to the amount of text they can deal with, so creating as small chunks of text as possible is necessary.
   - Embed text: this involves creating a numerical embedding for each chunk of text. 
   - Load embeddings to vectorstore: this involves putting embeddings and documents into a vectorstore(Native vector database or graph database in our case).
   - ![RAG Ingestion Diagram](./assets/RAG-ingestion.png)
 - Querying of Data
   - Use the user query to calculate vector embedding.
   - Lookup relevant documents: Using the embeddings and vectorstore created during ingestion, we can look up relevant documents for the answer
   - Generate a response: Given the user query and the relevant documents as context, we can use a language model to generate a response.
   - ![RAG Query Diagram](./assets/RAG-query.png)

### Re-Ranking

Reranking is a part of two-stage retreival systems where:
 1. Using vector databases and embedding model, we retrieve a set of relevant documents.
 2. Reranker model is used to "rerank" the documents retrieved in the first stage, and then we cut-off the context at top `k` results.

#### [ColBERT](https://github.com/stanford-futuredata/ColBERT) based Reranking

ColBERT encodes each passage into a matrix of token-level embeddings. Then at search time, it embeds every query into another matrix (shown in green) and efficiently finds passages that contextually match the query using scalable vector-similarity (`MaxSim`) operators.
Each passage(or chunk) is assgined a ColBERT score based upon similarity to the user query, and the score can be used to "rerank" chunks retrieved by vector search.

## How does Knowledge Graph fit in our AMP?

We leverage KG in two ways in order to make this RAG system better than plain(vanilla) RAG:
 1. We aim to enhance the quality of context retreived by choosing chunks from relatively "high-quality" papers.
 2. Provide additional information about the papers used to answer a certain question, which could have been more complex in case of traditional vector databases.

### Hybrid RAG

Since we have a small but related set of AI/ML papers, there would be a lot of "citation" relationships between papers. We define a paper to be of **"Higher Quality"** if it has more number of citations. The number of citations can be computed for a specific paper from the knowledge graph that we have built.

We employ a "hybrid" strategy to retrieve chunks where we take into consideration the semantic similarity as well as the "quality" of the paper the chunk is coming from, before passing it to LLM as context.

#### Hybrid retrieval algorithm for top `k` chunks:
 1. Retrieve `4*k` chunks using vector similarity(to the user query) from the Database.
 2. Rerank the chunks using [ColBERT](#colbert-based-reranking), cut-off the number of chunks at `2*k`. Store the **ColBERT Score** as well.
 3. Calculate a **hybrid score** = `(normalized ColBERT score) + (normalised number of citations to the chunk's paper)`. Rerank again based on the hybris score, and pick top `k` chunks as context.

### Additional Information for papers used.



## AMP Flow

## AMP Requirements
