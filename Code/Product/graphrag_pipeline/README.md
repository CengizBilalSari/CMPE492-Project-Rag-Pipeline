# GraphRAG Pipeline (Neo4j Edition)

A modular and resumable GraphRAG (Graph Retrieval-Augmented Generation) pipeline using Neo4j as the graph store. This package enables the extraction of entities and relationships from documents, community detection, and multiple search modes (Global, Local, PPR, and Lazy).

## Features

- **Modular Pipeline**: Discrete steps for chunking, extraction, resolution, embedding, community detection, and summarization.
- **Resumable Execution**: Skips already completed steps, making it efficient for large-scale document processing.
- **Multiple Search Modes**: Supports thematic (Global) and entity-focused (Local) searches, along with Graph-Reranked (PPR) and multi-step (Lazy) retrieval.
- **Neo4j Integration**: High-performance graph storage and querying.
- **Multi-LLM Support**: Works with OpenAI, Groq, and Google Vertex AI.

## Installation & Setup

1. **Environment Variables**: Create a `.env` file in the package root with the following credentials:

   ```env
   # Neo4j
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password

   # LLM Providers (Pick at least one)
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key
   ```

2. **Dependencies**: Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **NLP Model**: For Lazy Search, download the SpaCy model:

   ```bash
   python -m spacy download en_core_web_lg
   ```

## Configuration (`config.yaml`)

The `config.yaml` file controls all aspects of the pipeline. Below are the key options:

### `chunking`
- `strategy`: Method to split text. Options: `recursive` (default), `sentence`, `token`, `character`, `semantic`, `propositional`.
- `chunk_size`: Max characters or tokens per chunk (default: 1000).
- `chunk_overlap`: Overlap between chunks (default: 200).

### `llm`
- `provider`: LLM API to use. Options: `openai` (default), `groq`, `vertex`, `ollama`, `vllm`.
- `model`: Specific model name (e.g., `gpt-4o-mini`, `llama-3.1-70b-versatile`).
- `temperature`: Creativity level (default: 0.0 for deterministic output).
- `max_tokens`: Maximum response length.

### `embedding`
- `model`: Sentence-transformers model for vector embeddings (default: `all-MiniLM-L6-v2`).

### `entity_resolution`
- `enabled`: Whether to merge duplicate entities (default: `true`).
- `use_llm`: Use LLM for final merging decisions (default: `true`).
- `similarity_threshold`: Vector similarity threshold (0.0 to 1.0).
- `k_neighbors`: Number of neighbors to check for similarity.

### `community_detection`
- `algorithm`: Clustering algorithm. Options: `louvain` (default), `leiden` (requires GDS).
- `resolution`: Community granularity (default: 1.0).

### `search` (Global, Local, Lazy)
- `top_k`: Number of results to retrieve.
- `max_concurrency`: Parallel requests for summarization.
- `hop_depth`: (Local) Distance to traverse from entities.
- `max_subqueries`: (Lazy) Number of decomposition steps.

---

## Usage

### 1. Building the Graph (`graph_generation.py`)

Process documents to populate the Neo4j database. It automatically skips documents or steps that are already completed.

```bash
# Process a single document
python graph_generation.py --document path/to/report.pdf

# Process an entire directory of documents
python graph_generation.py --directory path/to/docs/
```

- **Modular Steps**: The pipeline runs in stages: Chunking -> Extraction -> Resolution -> Embedding -> Communities -> Summaries.
- **Retry Logic**: If interrupted, simply re-run the command; it will pick up where it left off.

### 2. Searching the Graph (`search_cli.py`)

Query your graph database using various search strategies.

```bash
# Global Search (Thematic, summarizes across the whole graph)
python search_cli.py "What are the main risks mentioned in the reports?" --mode global

# Local Search (Entity-focused, deep dive into specific nodes)
python search_cli.py "Tell me about Project X's timeline." --mode local

# Lazy Search (Decomposes complex queries into sub-questions)
python search_cli.py "Compare the financial performance of Dept A and Dept B." --mode lazy

# PPR (PageRank Reranked, uses graph topology to prioritize results)
python search_cli.py "Find key stakeholders for the acquisition." --mode ppr
```

#### Search Modes:
- **`global`**: Best for high-level "what" or "summarize" questions.
- **`local`**: Best for "who", "where", or specific entity details.
- **`lazy`**: Best for complex reasoning or multi-part questions.
- **`ppr`**: Uses Personalised PageRank to find conceptually relevant entities beyond direct neighbors.
