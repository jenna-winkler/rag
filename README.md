# Agentic RAG Example

An agentic RAG (Retrieval-Augmented Generation) system that lets you upload documents, ask questions about them, and get answers grounded in the document content.

Upload a PDF, Word doc, spreadsheet, image, or other supported file — the agent parses it, stores the embeddings in a vector database, and uses them to answer your questions in a multi-turn conversation.

### Built with

* **[BeeAI Framework](https://github.com/i-am-bee/beeai-framework)** – agent orchestration and reasoning
* **[Docling](https://github.com/docling-project/docling)** – document parsing and text extraction
* **[Milvus](https://milvus.io/)** – vector database for semantic search
* **[Arize Phoenix](https://phoenix.arize.com/)** – observability and tracing
* **[Agent Stack](https://agentstack.beeai.dev/)** – local platform runtime

### Supported file types

PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, HTML, CSV, PNG, JPEG, TIFF, BMP, WEBP

---

## How it works

1. **User uploads a document** — The file is sent to the agent through the Agent Stack UI.
2. **Docling extracts text** — The document is parsed asynchronously via the Docling service and split into chunks.
3. **Embeddings stored in Milvus** — Each chunk is embedded (using OpenAI `text-embedding-3-small` by default) and stored in a Milvus collection.
4. **User asks a question** — The BeeAI agent searches the vector store for relevant chunks and reasons over them to produce an answer.
5. **Conversation continues** — The agent maintains conversation history, so you can ask follow-up questions against the same documents.

---

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A [Milvus](https://milvus.io/) cloud instance (free tier available)

---

## Setup

### 1. Install Agent Stack

```bash
sh -c "$(curl -LsSf https://raw.githubusercontent.com/i-am-bee/agentstack/install/install.sh)"
```

This installs the `agentstack` CLI, downloads the platform, and launches the UI. For other options, see the [Quickstart guide](https://agentstack.beeai.dev/stable/introduction/quickstart).

### 2. Start the platform with Docling and Phoenix enabled

```bash
agentstack platform stop
agentstack platform start --set phoenix.enabled=true --set docling.enabled=true
```

### 3. Configure an LLM provider

```bash
agentstack model setup
```

Follow the prompts to connect an LLM provider (e.g., OpenAI). This is used for both chat completions and embeddings.

### 4. Create a Milvus cloud instance

1. Go to [milvus.io](https://milvus.io/) and click **Try Managed Milvus**
2. Create a project and cluster
3. Copy the **Public Endpoint** and **Token**
4. Register them with Agent Stack:

```bash
agentstack env add 'RAG Milvus' \
  MILVUS_DB_URI="your-endpoint" \
  MILVUS_DB_TOKEN="your-token"
```

### 5. Clone and install

```bash
git clone https://github.com/jenna-winkler/rag.git
cd rag
uv sync
```

---

## Running the agent

Start the agent server:

```bash
uv run server
```

Then open the Agent Stack UI:

```bash
agentstack ui
```

Go to [http://localhost:8334](http://localhost:8334), select the **RAG Milvus** agent, upload a document, and start asking questions.

---

## Observability

Traces are collected automatically via Arize Phoenix. Open [http://localhost:6006](http://localhost:6006) to explore agent runs, tool calls, and latency data.
