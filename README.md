# Graph RAG with Haystack and Neo4j

Extract structured knowledge graphs from unstructured text using [Haystack](https://haystack.deepset.ai/) pipelines and store them in [Neo4j](https://neo4j.com/).

Two implementations are provided:

| Script | Model | Requires |
|--------|-------|---------|
| `doc2graph_openai.py` | GPT-4o-mini (OpenAI API) | `OPENAI_API_KEY` |
| `doc2graph_local.py` | [Phi-3-mini-4k-instruct-graph](https://huggingface.co/EmergentMethods/Phi-3-mini-4k-instruct-graph) (local) | `HF_API_TOKEN` |

## How it works

```
Text documents
      │
      ▼
ChatPromptBuilder  ←  entity/relationship extraction prompt
      │
      ▼
LLM (OpenAI or local Phi-3)
      │
      ▼
JSON: { nodes: [...], edges: [...] }
      │
      ▼
Global ID deduplication  (cross-document entity resolution)
      │
      ▼
Neo4j  (MERGE — safe to re-run)
```

Each document is processed independently; entities with the same name across documents are merged into a single global node.

## Setup

### 1. Start Neo4j

```bash
docker compose up -d
```

Neo4j browser is available at http://localhost:7474 (credentials: `neo4j` / `password`).

### 2. Download the dataset

```bash
wget https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# edit .env and fill in your API keys
```

## Usage

### OpenAI (cloud)

```bash
export OPENAI_API_KEY=sk-...
python doc2graph_openai.py
```

### Local Phi-3 (offline)

```bash
export HF_API_TOKEN=hf_...
python doc2graph_local.py
```

The local pipeline runs on CPU by default. To use Apple Silicon GPU, uncomment the `Device.mps()` line in `doc2graph_local.py`.

## Exploring the graph

Once ingested, open the Neo4j browser at http://localhost:7474 and run:

```cypher
// View the full graph
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100

// Find all entities of a given type
MATCH (n:Node {type: "ORGANIZATION"}) RETURN n

// Explore relationships for a specific entity
MATCH (n:Node {name: "BBC"})-[r]-(m) RETURN n, r, m
```

## Project structure

```
.
├── doc2graph_openai.py   # OpenAI-based extraction pipeline
├── doc2graph_local.py    # Local Phi-3-based extraction pipeline
├── prompts.py            # Prompt templates (Microsoft GraphRAG + Phi-3 variants)
├── docker-compose.yml    # Neo4j service
├── requirements.txt
└── .env.example
```
