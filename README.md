# 🌿 Chat-ISV: Intelligent Q&A System for VOCs in the Iron and Steel Industry

[![Gradio](https://img.shields.io/badge/Gradio-6.9.0-blue?logo=gradio)](https://gradio.app)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green?logo=langchain)](https://python.langchain.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-orange?logo=neo4j)](https://neo4j.com)

**Chat-ISV** is an intelligent question-answering system specifically designed for the iron and steel industry, focusing on **Volatile Organic Compounds (VOCs)** emission sources, control technologies, and regulatory compliance. It combines **knowledge graph reasoning**, **vector retrieval**, and **LLM-powered synthesis** to provide authoritative, traceable answers.

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-3-Tier-blue" alt="3-Tier Architecture">
  <img src="https://img.shields.io/badge/Features-Knowledge%20Graph%20%2B%20Vector%20DB-green" alt="Hybrid RAG">
</p>

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Neo4j Knowledge Graph** | Industry-specific VOCs graph with 9 entity types (Process, EmissionSource, VOCSpecies, ControlTech, Method, Regulation, Factor, Mechanism, Scenario) and 12 relation types |
| **3-Tier Hybrid RAG** | Graph RAG → Local Vector DB (FAISS/BGE) → Wikipedia fallback for robust recall |
| **Dynamic Subgraph Visualization** | PyVis-powered interactive network visualization showing 1st-degree relations |
| **Smart Query Translation** | LLM-powered Cypher query generation with domain-specific rules (fuzzy matching, type casting, context isolation) |
| **Context-Aware Conversations** | Multi-turn support with previous context preservation |
| **Professional Answer Synthesis** | Engineering-focused LLM prompts for authoritative, unit-preserving, hierarchical responses |

---

## 🏗️ Architecture

### 3-Tier Retrieval Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              User Question                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Tier 1: Neo4j Knowledge Graph RAG                      │
│  • Graph schema-aware Cypher generation                                      │
│  • Fuzzy matching (toLower/toString) for robust entity retrieval            │
│  • Dynamic subgraph visualization via PyVis                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                  No results found?  │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│              Tier 2: Local Vector Database (FAISS + BGE)                   │
│  • High-dimensional semantic search (8 retrievals)                           │
│  • Local embedding model (CPU mode supported)                                │
│  • LLM-based relevance filtering                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                  No relevant docs?  │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   Tier 3: Wikipedia API Fallback                           │
│  • Global knowledge coverage for common VOCs concepts                        │
│  • Summarization for concise answers                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Knowledge Graph Schema

**Entity Types (9):**
- `Process` - Industrial processes (sintering, coking, annealing, etc.)
- `EmissionSource` - Emission sources within processes
- `VOCSpecies` - Specific volatile organic compounds (CH4, benzene, toluene, etc.)
- `ControlTech` - Control technologies (RTO, SCR, activated carbon, etc.)
- `Method` - Measurement/observation methods
- `Regulation` - Environmental regulations
- `Factor`, `Mechanism`, `Scenario` - Supporting technical concepts

**Relationship Types (12):**
- `emits`: EmissionSource/Process → VOCSpecies
- `controlled_by`: Process/EmissionSource → ControlTech
- `measured_by`: EmissionSource/VOCSpecies → Method
- `regulated_by`: Process/EmissionSource → Regulation
- `participates_in`, `influenced_by`, `correlates_with` - Other associations

---

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9+
- **Neo4j**: 4.4+ (for knowledge graph)
- **OpenAI API**: GPT-4o-mini or compatible (for query generation & answer synthesis)

### Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Data Processing Pipeline

This project follows a modular design with three sequential steps:

#### 1. PDF Information Extraction (`get_data_fromPDF/`)

Extract entities and relations from PDF documents:

```bash
cd get_data_fromPDF
# 1. Edit config.json to set your API key and input directory
# 2. Place PDF files in the input directory
python llm_extra_pdf.py
```

Output is JSONL format containing extracted entities, relations, and evidence text.

#### 2. Knowledge Graph Construction (`data_to_kg/`)

Import extracted data into Neo4j:

```bash
cd data_to_kg
# 1. Edit to_4demo.py to set NEO4J_URI, NEO4J_USER, NEO4J_PASS
# 2. Set JSONL_PATH to point to the output from Step 1
python to_4demo.py
```

#### 3. Vector Database Construction (Optional)

Build local vector retrieval with FAISS:

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Load documents
loader = TextLoader("path/to/documents")
documents = loader.load()

# Create embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Build and save FAISS index
faiss = FAISS.from_documents(documents, hf_embeddings)
faiss.save_local("test_data/env_faiss_index")
```

---

## 🔧 Configuration

### LLM API Config (`get_data_fromPDF/config.json`)

```json
{
  "api_key": "your-api-key",
  "api_base": "https://api.openai.com/v1",
  "model": "gpt-4o-mini",
  "input_dir": "path/to/pdfs",
  "output_jsonl": "output.jsonl",
  "chunk_chars": 2000,
  "max_retry": 6,
  "temperature": 0.2
}
```

### Neo4j Config (`data_to_kg/to_4demo.py`)

```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "your-password"
BATCH_SIZE = 200
```

### Cypher Query Rules

Key rules embedded in prompts:

1. **Strict relationship types only** - No invented relation types
2. **Fuzzy matching** - Use `toLower(toString(n.name)) CONTAINS`
3. **Avoid over-constraining** - Don't chain multiple AND conditions on names
4. **Clean output** - Return only name and details, not entire nodes
5. **Single statement** - Exactly one Cypher query per request
6. **Context isolation** - Ignore "--- PREVIOUS CONTEXT ---" in queries

---

## 📂 Project Structure

```
Chat-ISV/
├── data_to_kg/                # Knowledge Graph Construction
│   └── to_4demo.py           # Neo4j data ingestion script
├── get_data_fromPDF/          # PDF Information Extraction
│   ├── llm_extra_pdf.py      # PDF extraction + LLM entity-relation extraction
│   ├── prompts.py            # LLM extraction prompts
│   └── progress.json         # Extraction progress checkpoint
├── test_data/                 # Test Data
│   ├── env_faiss_index/      # Vector index files
│   ├── score_by_expert.xlsx  # Expert scoring data
│   └── Visualization of scoring.py
├── text_record/               # Text Records
│   └── text.log              # Run logs
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🎨 Customization

### Adding New Entity Types

1. Update `ENTITY_LABELS` in `data_to_kg/to_4demo.py`
2. Update output schema in `get_data_fromPDF/prompts.py`
3. Update color map in visualization (if needed)

### Modifying Retrieval Strategy

- Adjust `top_k` in graph query
- Change vector search k value
- Modify Wikipedia fallback logic

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Graph connection failed | Verify Neo4J_URI, username, password; check database exists |
| Vector DB load failed | Ensure `index.faiss` and `index.pkl` exist in specified path |
| API key validation error | Key must start with "sk-"; check base_url for compatible APIs |
| No results in graph tier | Try more general terms; verify graph schema matches queries |
| Slow responses | Consider GPU for embeddings; adjust temperature parameter |

---

