# Aarav Local RAG Engine

A fully local Retrieval-Augmented Generation (RAG) system. Upload PDF documents or paste raw text, and then ask natural language questions about that content. All processing — embeddings, vector search, and LLM inference — runs on your machine with no cloud API calls.

## What It Does

1. **Ingest** — Upload a PDF or paste text into the UI. The backend splits the content into overlapping 500-word chunks, generates a 768-dimensional embedding for each chunk using `nomic-embed-text` via Ollama, and stores the vectors in a PostgreSQL database powered by the `pgvector` extension.
2. **Query** — Ask a question in the chat interface. The question is embedded and compared against all stored vectors using cosine distance. The top 3 most relevant chunks are retrieved and passed as context to `llama3.2`, which generates a grounded answer.

## Requirements

- Python 3.10+ (project optimized for 3.14)
- [PostgreSQL](https://www.postgresql.org/) running locally on port `5432`
- [Ollama](https://ollama.com/) installed (the backend will attempt to start it automatically if it is not running)

## Setup

### 1. PostgreSQL Database

Create the database and enable the `pgvector` extension:

```sql
CREATE DATABASE aarav_vector_db;
\c aarav_vector_db
CREATE EXTENSION IF NOT EXISTS vector;
```

The `document_store` table is created automatically by the backend on first run.

### 2. Python Environment

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install fastapi uvicorn psycopg2-binary pgvector ollama pypdf streamlit requests
```

### 3. Ollama Models

The backend pulls these automatically if they are missing, but you can pre-fetch them manually:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Running the Project

Open two terminal windows with the virtual environment activated in both.

**Terminal 1 — Start the backend API:**

```bash
python main.py
```

The FastAPI server starts at `http://127.0.0.1:8000`.

**Terminal 2 — Start the frontend UI:**

```bash
streamlit run ui.py
```

The Streamlit app opens in your browser at `http://localhost:8501`.

## Usage

1. Navigate to **Document Ingestion Workspace** in the sidebar.
2. Upload a PDF using the file uploader on the left, or paste text directly into the text box on the right.
3. Click **Compile & Generate Embeddings** to process and store the content into the `aarav_vector_db` cluster.
4. Switch to **Conversational Chat Pipeline** and ask questions about the ingested content.

## Architecture

| Component | Role |
|---|---|
| Streamlit (`ui.py`) | Browser-based frontend for ingestion and chat |
| FastAPI (`main.py`) | REST API backend handling ingestion and query logic |
| PostgreSQL + pgvector | Persistent vector storage and cosine similarity search |
| Ollama + nomic-embed-text | Local text embedding (768-dimensional vectors) |
| Ollama + llama3.2 | Local LLM for answer generation |

See `rag_architecture.typ` for a detailed technical breakdown of the pipeline.

### System Topology
```mermaid
graph TD
    subgraph Frontend [User Interface Layer]
        UI[Streamlit Application ui.py]
    end

    subgraph Backend [Local REST Engine Layer]
        API[FastAPI Server main.py]
        Ollama[Ollama API Client]
    end

    subgraph Infrastructure [Persistent Storage & Hardware]
        DB[(PostgreSQL + pgvector <br> aarav_vector_db)]
        Daemon[Ollama Service Daemon]
        Models[Local Weights: <br> llama3.2 & nomic-embed-text]
    end

    UI -- HTTP POST /ingest or /query --> API
    API -- Psycopg2 Driver --> DB
    API -- Localhost Socket:11434 --> Ollama
    Ollama --> Daemon
    Daemon --> Models
```

### Data Ingestion Pipeline Flow
```mermaid
graph TD
    Start([User Drops PDF or Pastes Text & Clicks Compile]) --> SourceType{Input Source?}
    
    SourceType -- PDF --> Extract[Extract Raw Character Streams via PyPDF]
    Extract --> Condition{Is Text Extracted?}
    Condition -- No --> Error[Throw UI Error: Blank or Scan File] --> End([Terminate Process])
    Condition -- Yes --> Chunk

    SourceType -- Raw Text --> Chunk[Split Text into 500-Word Windows with Overlap]
    
    Chunk --> LoopStart[For Each Individual Chunk]
    LoopStart --> Embed[Request Spatial Coordinates via nomic-embed-text]
    Embed --> SQL[Execute INSERT with Explicit Array Cast]
    SQL --> LoopCondition{More Chunks?}
    
    LoopCondition -- Yes --> LoopStart
    LoopCondition -- No --> Commit[Execute DB Commit & Return 200 OK]
    Commit --> Success[Show Ingestion Success Toast in UI] --> End
```

### Execution Sequence (Real-Time RAG Chat Loop)
```mermaid
sequenceDiagram
    autonumber
    actor User as User Terminal
    participant UI as Streamlit UI
    participant API as FastAPI (main.py)
    participant DB as pgvector (aarav_vector_db)
    participant OLL as Ollama Engine

    User->>UI: Types Question & Press Enter
    UI->>API: HTTP POST /query {prompt: "..."}
    
    Note over API,OLL: Calculate spatial query vector
    API->>OLL: ollama.embed(nomic-embed-text, prompt)
    OLL-->>API: Returns 768-Dimension Float Array
    
    Note over API,DB: Mathematical Distance Check
    API->>DB: SQL SELECT Ordered By embedding <=> query_vector::vector LIMIT 3
    DB-->>API: Returns Top 3 Raw Context Text Blocks
    
    Note over API,OLL: Construct System Bounded Prompt
    API->>OLL: ollama.chat(llama3.2, system: context + user: prompt)
    OLL-->>API: Returns Compiled Natural Language Answer
    
    API-->>UI: JSON Payload Response {"answer": "..."}
    UI->>User: Appends Frame to Chat History Logs
```
