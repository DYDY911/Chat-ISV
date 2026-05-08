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
| **Context-Aware Conversations** | Multi-turn support with previous context preservation and asymmetric mapping handling |
| **Professional Answer Synthesis** | Engineering-focused LLM prompts for authoritative, unit-preserving, hierarchical responses |

---

## 🏗️ Architecture

### 3-Tier Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Question                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Tier 1: Neo4j Knowledge Graph RAG                      │
│  • Graph schema-aware Cypher generation                                     │
│  • Fuzzy matching (toLower/toString) for robust entity retrieval            │
│  • Dynamic subgraph visualization via PyVis                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                  No results found?  │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Tier 2: Local Vector Database (FAISS + BGE)                   │
│  • High-dimensional semantic search (8 retrievals)                          │
│  • Local embedding model (CPU mode supported)                               │
│  • LLM-based relevance filtering                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                  No relevant docs?  │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Tier 3: Wikipedia API Fallback                           │
│  • Global knowledge coverage for common VOCs concepts                       │
│  • Summarization for concise answers                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Knowledge Graph Schema

**Entity Types:**
- `Process` - Industrial processes (sintering, coking, annealing, etc.)
- `EmissionSource` - Emission sources within processes
- `VOCSpecies` - Specific volatile organic compounds (CH4, benzene, toluene, etc.)
- `ControlTech` - Control technologies (RTO, SCR, activated carbon, etc.)
- `Method` - Measurement/observation methods
- `Regulation` - Environmental regulations
- `Factor`, `Mechanism`, `Scenario` - Supporting technical concepts

**Key Relations:**
- `emits`: EmissionSource/Process → VOCSpecies
- `controlled_by`: Process/EmissionSource → ControlTech
- `measured_by`: EmissionSource/VOCSpecies → Method
- `participates_in`, `influenced_by`, `correlates_with`

---

## 📋 Prerequisites

### Environment Requirements

- **Python**: 3.9+
- **Neo4j**: 4.4+ (for knowledge graph)
- **OpenAI API**: GPT-4o-mini or compatible (for query generation & answer synthesis)

### System Dependencies

| Dependency | Purpose |
|------------|---------|
| `gradio` | Web UI framework |
| `langchain` | RAG orchestration |
| `langchain-openai` | OpenAI integration |
| `neo4j` | Graph database driver |
| `faiss-cpu` or `faiss-gpu` | Vector search |
| `sentence-transformers` | BGE embeddings |
| `pyvis` | Graph visualization |
| `wikipedia` | Fallback knowledge source |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/Chat-ISV.git
cd Chat-ISV
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Neo4j Knowledge Graph

1. Start your Neo4j instance:
```bash
docker run -e NEO4J_AUTH=neo4j/password -p 7474:7474 -p 7687:7687 neo4j:4.4
```

2. Load data via `to_4demo.py` (adjust credentials in the file):
```bash
python to_4demo.py
```

### 4. Prepare Local Vector Database (Optional but Recommended)

Generate embeddings for your domain documents:

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

# Load your documents
# ... document loading code ...

# Create embeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create and save FAISS index
faiss = FAISS.from_documents(documents, hf_embeddings)
faiss.save_local(".")
```

This creates `index.faiss` and `index.pkl` in the root directory.

### 5. Configure Environment Variables

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or compatible endpoint
```

### 6. Launch the Application

```bash
python app0508.py
```

The UI will be available at `http://localhost:7860`

---

## 💬 Usage

### Input Examples

```
1. Which EmissionSources release Methane (CH4)?
2. List common VOCs ControlTech, excluding laboratory equipment.
3. What is the basic concept of Volatile Organic Compounds (VOCs)?
4. What Methods in the graph can be used to observe VOCsSpecies?
5. What specific VOC species are emitted from the iron and steel industry?
```

### Workflow

1. **Enter API Key**: Provide a valid OpenAI API key (required for LLM queries)
2. **Ask Question**: Type your question in the chat input
3. **View Response**: The system displays:
   - Tier 1: Graph query progress (🧠 [1/3])
   - Tier 2: Vector DB search if graph missed (📖 [2/3])
   - Tier 3: Wikipedia fallback if needed (🌐 [3/3])
4. **Explore Visualization**: The subgraph network shows related entities

---

## 🔧 Configuration

### Key Variables (in `app0508.py`)

```python
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE_NAME = "e554e789"  # Adjust to your database name
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
```

### Cypher Query Rules (Critical)

The system uses specialized prompts with these rules:

1. **Strict relationships only** - No invented relation types
2. **Fuzzy matching** - Always use `toLower(toString(n.name)) CONTAINS`
3. **Avoid over-constraining** - Don't chain multiple AND conditions on names
4. **Clean dictionaries** - Return only name and details, not entire nodes
5. **Single statement** - Exactly one Cypher query per request
6. **Context isolation** - Ignore "--- PREVIOUS CONTEXT ---" in queries
7. **No post-processing** - Let the QA model handle empty states

---

## 📂 Project Structure

```
Chat-ISV/
├── app0508.py          # Main application - Gradio UI + 3-tier RAG
├── prompts.py          # LLM prompts for extraction & QA
├── llm_extra_pdf.py    # PDF extraction + knowledge graph population
├── to_4demo.py         # Neo4j ingestion script
├── requirements.txt    # Python dependencies
├── progress.json       # Ingestion checkpoint (auto-generated)
├── config.json         # LLM API configuration (create from config.json.example)
├── index.faiss         # FAISS vector index (auto-generated)
└── index.pkl           # FAISS index metadata (auto-generated)
```

---

## 🎨 Customization

### Adding New Entity Types

1. Update `ENTITY_LABELS` in `to_4demo.py`
2. Add labels to `prompts.py` output schema
3. Update color map in `app0508.py` `generate_vis_subgraph_html`

### Modifying Retrieval Strategy

- Adjust `top_k` in `GraphCypherQAChain` (line 153)
- Change vector search k in `answer_question` (line 334)
- Modify Wikipedia calls in Tier 3 logic (lines 379-390)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Graph connection failed | Verify Neo4J_URI, username, password; check database exists |
| Vector DB load failed | Ensure `index.faiss` and `index.pkl` exist in root directory |
| API key validation error | Key must start with "sk-"; check base_url if using compatible API |
| No results in graph tier | Try more general terms; check graph schema matches queries |
| Slow responses | Consider GPU for embeddings; increase temperature for more results |

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **LangChain** for RAG orchestration framework
- **Neo4j** for graph database capabilities
- **HuggingFace BGE** for embedding models
- **Gradio** for rapid UI development

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

*Built with ❤️ for the environmental engineering and steel industry communities*
