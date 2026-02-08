# GraphRAG: A Deep Technical Overview

> **Graph-based Retrieval Augmented Generation** - Microsoft Research's approach to unlocking LLM discovery on narrative private data.

[![Paper](https://img.shields.io/badge/arXiv-2404.16130-red)](https://arxiv.org/pdf/2404.16130)
[![Blog](https://img.shields.io/badge/Microsoft-Research%20Blog-blue)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Why GraphRAG?](#why-graphrag)
- [System Architecture](#system-architecture)
- [Phase 1: Indexing Pipeline](#phase-1-indexing-pipeline)
- [Phase 2: Query Engine](#phase-2-query-engine)
- [Global Search Deep Dive](#global-search-deep-dive)
- [Key Design Decisions](#key-design-decisions)

---

## Introduction

GraphRAG is a structured, hierarchical approach to Retrieval Augmented Generation (RAG) that uses **LLM-generated knowledge graphs** to significantly improve question-answering capabilities on private datasets. Unlike traditional vector-based RAG systems, GraphRAG can:

- Connect disparate pieces of information across documents
- Answer holistic questions about entire datasets
- Provide provenance and source grounding for generated answers
- Support **self-reflection** where the LLM validates its own responses

The data is organized **hierarchically**, enabling both:
- **General to Specific**: Top-down exploration from themes to details
- **Specific to General**: Bottom-up aggregation from entities to patterns

---

## Why GraphRAG?

Traditional Baseline RAG fails in two critical scenarios:

| Scenario | Example Query | Why Baseline RAG Fails |
|----------|--------------|----------------------|
| **Connecting the Dots** | "What has Novorossiya done?" | Vector search can't traverse relationships between entities |
| **Holistic Understanding** | "What are the main themes in this dataset?" | No single text chunk contains dataset-wide themes |

GraphRAG solves these by building a **knowledge graph** with **hierarchical community structure**, enabling both local entity reasoning and global dataset understanding.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            GraphRAG SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PHASE 1: INDEXING                               │   │
│   │                                                                     │   │
│   │   Documents → Chunks → Entities/Relations → Graph → Communities     │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     PHASE 2: QUERY                                  │   │
│   │                                                                     │   │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
│   │   │  Global  │ │  Local   │ │  DRIFT   │ │  Basic   │               │   │
│   │   │  Search  │ │  Search  │ │  Search  │ │  Search  │               │   │
│   │   └──────────┘ └──────────┘ └──────────┘ └──────────┘               │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Indexing Pipeline

The indexing pipeline transforms raw documents into a queryable knowledge structure through 5 stages:

### Stage A: Text Chunking

Documents are split into **TextUnits** - analyzable chunks that serve as atomic units for extraction.

```
Document (50,000 tokens)
         │
         ▼
┌─────────────────────────────────────────────┐
│  Chunk 1   │  Chunk 2   │  ...  │  Chunk N  │
│  600 tok   │  600 tok   │       │  600 tok  │
└─────────────────────────────────────────────┘
```

- **Default chunk size**: ~600 tokens (configurable)
- Smaller chunks = higher fidelity extraction
- Larger chunks = faster processing, lower cost

### Stage B: Entity, Relationship & Claims Extraction

The LLM analyzes each chunk to extract structured information:

| Extraction Type | Description | Example |
|----------------|-------------|---------|
| **Entities** | People, places, organizations, events | "Donald Trump", "New York", "Tesla" |
| **Relationships** | Connections between entities | Trump → FOUNDED → Trump Organization |
| **Claims** | Time-bound factual statements | "Trump became CEO of Trump Org in 1971" |

Claims capture the **who, what, when, where** with temporal bounds and evaluated status.

### Stage C: Knowledge Graph Construction

Individual subgraphs from each chunk are merged into a unified knowledge graph:

```
Chunk 1: [Trump] ──works_at──▶ [Trump Org]
Chunk 2: [Trump] ──founded──▶ [Trump Org]
Chunk 3: [Trump] ──lives_in──▶ [NYC]
                     │
                     ▼ MERGE
         ┌───────────────────────────┐
         │    Unified Knowledge      │
         │         Graph             │
         │                           │
         │  [Trump]──┬──works_at──┐  │
         │           ├──founded───┼─▶[Trump Org]
         │           └──lives_in──┘  │
         │                  │        │
         │                  ▼        │
         │               [NYC]       │
         └───────────────────────────┘
```

**Key merging rules:**
- Entities with same name/type → merged into single node
- Multiple descriptions → summarized by LLM
- Edge weights → based on relationship **frequency** across chunks

### Stage D: Community Detection

The **Leiden algorithm** performs hierarchical clustering on the graph:

```
                     ┌────────────────────┐
                     │     Level 0        │  ◀── Entire graph (1 community)
                     │   "All Entities"   │
                     └─────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
        │  Level 1  │    │  Level 1  │    │  Level 1  │
        │ "Politics"│    │ "Business"│    │ "Sports"  │
        └─────┬─────┘    └─────┬─────┘    └───────────┘
              │                │
        ┌─────┴─────┐    ┌─────┴─────┐
        │           │    │           │
   ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
   │Level 2  │ │Level 2  │ │Level 2  │
   │"US Pol" │ │"EU Pol" │ │"Finance"│
   └─────────┘ └─────────┘ └─────────┘
```

This hierarchical structure enables queries at different **granularity levels**.

### Stage E: Community Summarization

Each community receives an LLM-generated summary report:

```markdown
## Community: US Political Entities

### Executive Summary
This community encompasses key political figures and organizations
involved in US domestic policy...

### Key Entities
- Donald Trump (Person) - 45th President
- Joe Biden (Person) - 46th President  
- White House (Place) - Executive residence

### Key Findings
1. Strong relationship network between political figures and lobbyists
2. Recurring themes of policy disputes on economic matters
3. [Data: Reports (12, 45, 67)]
```

**Bottom-up summarization**: Lower-level community summaries are passed up to inform higher-level summaries, creating coherent hierarchical understanding.

---

## Phase 2: Query Engine

GraphRAG supports 4 distinct query modes:

### 1. Global Search 🌐

**Purpose**: Answer holistic questions about the entire dataset

| Question Type | Example |
|--------------|---------|
| Thematic | "What are the main themes in this dataset?" |
| Aggregate | "What are the most common conflict patterns?" |
| Summary | "Give me an overview of all political events" |

Uses **Map-Reduce** pattern with community summaries. [See detailed explanation below](#global-search-deep-dive).

### 2. Local Search 🔍

**Purpose**: Detailed questions about specific entities

| Question Type | Example |
|--------------|---------|
| Entity-specific | "When and where did Donald Trump enter business?" |
| Relationship | "What is the connection between Company X and Person Y?" |
| Timeline | "What events involved Entity Z in 2023?" |

**Mechanism**: Starts from query entities, fans out through graph relationships to gather relevant context.

### 3. DRIFT Search 🚀

**Purpose**: Enhanced local search with community context

DRIFT (Dynamic Reasoning and Inference with Flexible Traversal) adds community summaries to the context window, providing:
- Broader thematic context for entity-focused queries
- Better understanding of entity's role within communities
- Hybrid local + global perspective

### 4. Basic Search 📊

**Purpose**: Standard vector similarity search (Baseline RAG)

- Traditional top-k retrieval based on embedding similarity
- Useful when query is well-represented in individual text chunks
- Falls back to this for simple factual lookups

---

## Global Search Deep Dive

Global Search is the most sophisticated query mode, using a **Map-Reduce** approach to synthesize information across all communities.

### The Challenge

```
Problem: 500 community reports × 500 tokens = 250,000 tokens
         LLM context window = ~128,000 tokens
         
         ❌ Cannot fit all reports in one prompt!
```

### Solution: Map-Reduce with Batching

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GLOBAL SEARCH FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Step 1: CONTEXT BUILDING                                              │
│   ─────────────────────────                                             │
│                                                                         │
│   ┌─────────────────────────────────────────────────┐                   │
│   │     500 Community Reports                       │                   │
│   │  [R1][R2][R3]...[R100]...[R200]...[R500]        │                   │
│   └───────────────────────┬─────────────────────────┘                   │
│                           │                                             │
│                           ▼ random.shuffle()                            │
│                                                                         │
│   ┌─────────────────────────────────────────────────┐                   │
│   │  [R234][R12][R456][R89]...[R7][R301]...[R99]    │                   │
│   └───────────────────────┬─────────────────────────┘                   │
│                           │                                             │
│                           ▼ Split by token limit (8000 tokens/batch)    │
│                                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│   │ Batch 1  │  │ Batch 2  │  │ Batch 3  │  │ Batch N  │                │
│   │ ~50 reps │  │ ~50 reps │  │ ~50 reps │  │ ~50 reps │                │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
│        │             │             │             │                      │
│        └─────────────┴──────┬──────┴─────────────┘                      │
│                             │                                           │
│   Step 2: MAP (Parallel)    ▼                                           │
│   ──────────────────────────                                            │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  For each batch (in parallel via asyncio.gather):                │  │
│   │                                                                  │  │
│   │  Prompt: "Given these reports, answer the user's question.       │  │
│   │           Return key points with importance scores (0-100)."     │  │
│   │                                                                  │  │
│   │  Response format:                                                │  │
│   │  {                                                               │  │
│   │    "points": [                                                   │  │
│   │      {"description": "Theme 1... [Data: Reports (2,7)]", "score": 85},  │
│   │      {"description": "Theme 2... [Data: Reports (12)]", "score": 72}│
│   │    ]                                                             │  │
│   │  }                                                               │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                             │                                           │
│                             ▼                                           │
│   Step 3: FILTER & RANK                                                 │
│   ─────────────────────                                                 │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  1. Collect all points from all batches                          │  │
│   │  2. Filter out points with score = 0                             │  │
│   │  3. Sort by score DESCENDING                                     │  │
│   │  4. Take top points until max_data_tokens reached                │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                             │                                           │
│                             ▼                                           │
│   Step 4: REDUCE                                                        │
│   ──────────────                                                        │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Prompt: "Synthesize these analyst reports into a final answer"  │  │
│   │                                                                  │  │
│   │  ----Analyst 1----                                               │  │
│   │  Importance Score: 95                                            │  │
│   │  The primary theme is conflict resolution [Data: Reports (1,5)]  │  │
│   │                                                                  │  │
│   │  ----Analyst 2----                                               │  │
│   │  Importance Score: 88                                            │  │
│   │  Economic factors play a key role [Data: Reports (3,7)]          │  │
│   │  ...                                                             │  │
│   │                                                                  │  │
│   │  → LLM generates final comprehensive answer                      │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                             │                                           │
│                             ▼                                           │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  FINAL ANSWER with source citations [Data: Reports (...)]       │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Random Shuffle?

The shuffle before batching is a **critical design decision**:

| Without Shuffle | With Shuffle |
|----------------|--------------|
| Batch 1: All politics reports | Batch 1: Mixed topics |
| Batch 2: All economics reports | Batch 2: Mixed topics |
| Each batch has narrow perspective | Each batch has diverse perspective |
| Single topic failure = total loss | Redundancy across batches |
| Biased toward early communities | Fair representation |

**Code reference** (`community_context.py`):
```python
if shuffle_data:
    random.seed(random_state)  # random_state=86 for reproducibility
    random.shuffle(selected_reports)
```

### Importance Scoring

Each point from MAP phase includes a score (0-100):

- **0**: No relevant information ("I don't know")
- **1-30**: Tangentially related
- **31-70**: Moderately relevant
- **71-100**: Highly relevant, directly answers query

Only points with `score > 0` proceed to the REDUCE phase.

---

## Key Design Decisions

### 1. Hierarchical Data Organization

```
Level 0 (Coarse) ─────────────────────────────────► Level N (Fine)
     │                                                    │
     │  "What are the main themes?"                       │  "What did person X do on date Y?"
     │                                                    │
     └──── Use high-level community summaries             └──── Use detailed entity relationships
```

### 2. Self-Reflection / Validation

The LLM performs internal validation:
- Importance scoring acts as self-assessment
- Points with score=0 are filtered (LLM admits uncertainty)
- Source citations enable verification

### 3. Provenance & Grounding

Every claim includes source references:
```
"Person X is involved in controversy [Data: Reports (2, 7, 34)]"
```
This enables:
- Human verification of claims
- Traceability to original documents
- Reduced hallucination risk

### 4. Token Budget Management

```python
max_context_tokens = 8000  # Per batch in MAP phase
max_data_tokens = 8000     # For aggregated points in REDUCE phase
```

Careful token management ensures:
- Consistent batch sizes
- Predictable API costs
- Reliable response generation

---

## References

- **Paper**: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/pdf/2404.16130)
- **Blog**: [GraphRAG: Unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
- **Documentation**: [Microsoft GraphRAG Docs](https://microsoft.github.io/graphrag)

---

*This document provides a technical deep-dive into GraphRAG. For getting started, see the official [Quick Start Guide](https://microsoft.github.io/graphrag/get_started/).*
