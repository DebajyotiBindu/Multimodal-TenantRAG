# Multimodal-TenantRAG

An asynchronous, multi-tenant Multimodal RAG (Retrieval-Augmented Generation) pipeline engineered with LangChain, ChromaDB, and Groq. The system processes unstructured technical documents containing interleaved layout structures (text chunks and diagrams) and isolates them strictly per-session, utilizing an SQLite relational data layer to maintain a comprehensive, audit-ready transaction log.

## Key Architectural Features

- **Isolated Multi-Tenancy:** Implements runtime logical data isolation within shared ChromaDB collections using explicit custom metadata injection (`user_id`). This guarantees strict tenant data boundaries during similarity vector operations.
- **Dual-Path Multimodal Ingestion:** - **Text Branch:** Chunks document layouts using a `RecursiveCharacterTextSplitter` configured for hierarchical context retention.
  - **Visual Branch:** Identifies embedded figures, diagrams, and tables, and dynamically generates rich contextual visual summaries using the `meta-llama/llama-4-scout-17b-16e-instruct` vision architecture over a low-latency Groq client runtime channel.
- **Unified Semantic Space:** Vectorizes both layout text blocks and generated image descriptions using the high-fidelity `sentence-transformers/all-mpnet-base-v2` embedding model, allowing unified cross-medium vector retrieval.
- **Relational Auditing & Tracking:** Integrates an absolute transaction layer with SQLAlchemy (SQLite). Every query maps directly to a primary log, capturing the exact `user_id`, raw input, LLM inference text, and multiple line-item `RetrievedContext` blocks identifying exactly which text chunks or image file paths were served to ground the response.
- **Asynchronous Token Streaming:** Driven by a concurrent Chainlit runtime that decouples heavy synchronous CPU-bound vector matching from the main thread using an execution pool, delivering token-by-token streaming back to the client interface via a live websocket state engine.

---

## 🛠️ Tech Stack & Dependencies

- **Orchestration & Drivers:** `langchain`, `langchain-huggingface`, `langchain-groq`
- **Vector Indexing Engine:** `chromadb`, `langchain-chroma`
- **Relational ORM Mapping:** `sqlalchemy`
- **Compute Provider & LLM:** `groq` (Model: `qwen/qwen3-32b`)
- **Real-Time UI Socket Interface:** `chainlit`
- **Environment Management:** `python-dotenv`

---

## Project Structure

```text
├── COMPONENTS/
│   └── captions.py         # Encodes layout images to base64 and interfaces Groq Vision API
├── src/
│   ├── loaders.py          # Custom TextLoader and ImageLoader for extraction parsing
│   ├── splitting.py        # Recursive chunk partitioning and HuggingFace token embedding
|   ├── database.py          # Creates a chroma database in peristent directory and stores the chunks and their embeddings for retrieval
│   ├── audit_db.py         # SQLAlchemy relational schema models and transaction logger
│   └── retrieval.py        # Multi-tenant filtering vector search and LLM invocation routing
├── app.py                  # Core asynchronous Chainlit websocket app entry point
├── Dockerfile
├── Docker-compose.yaml
├── requirements.txt          
└── README.md               
```

# Technical Setup & Installation

## 1. Clone the Workspace
```
git clone https://github.com/DebajyotiBindu/Multimodal-TenantRAG.git
cd Multimodal-TenantRAG
```

## 2. Configure Local Environment Constraints
Create a .env file in the root directory and add your Groq API credentials:

Inside .env file-
```
GROQ_API_KEY=gsk_your_api_token_here
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="your-langsmith-api-key"
```

## 3. Verify Local File System Mount Paths
Open app.py and make sure the configurable absolute local runtime directory references align with your development environment drive mounting schema:

```
VECTOR_DB_DIR = "./Vector_Database"
STATIC_IMAGE_DIR = "./Static"
TEMP_DATA_DIR = "./data"
```

## 4. Execute the Application Instance
Boot the asynchronous engine over local websockets:
```
chainlit run app.py
```
