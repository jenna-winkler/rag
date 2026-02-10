# Agentic RAG Example

This example demonstrates an agentic RAG architecture built with:

* **BeeAI Framework** – agent orchestration and reasoning  
* **Docling** – document processing  
* **Milvus** – vector store for semantic retrieval  
* **Arize Phoenix** – observability and tracing  
* **Agent Stack** – local platform runtime and deployment

---

## Workflow

1. Request to Agent Stack – The system receives a new user message.

2. Extract Data with Docling – Relevant information is parsed and extracted from the document.

3. Store in Milvus – The extracted embeddings are stored in the Milvus vector database.

4. Run BeeAI Agent – The BeeAI agent performs reasoning and retrieves relevant context.

5. Store Conversation – The conversation and results are logged for observability.

6. Respond to User – The final response is generated and returned to the user.

---

## 1. Installation

Install Agent Stack using the official one-line installer:

```bash
sh -c "$(curl -LsSf https://raw.githubusercontent.com/i-am-bee/agentstack/install/install.sh)"
```

This will:
- Install the `agentstack` CLI  
- Download and start the Agent Stack platform  
- Prompt you to configure an LLM provider  
- Launch the Agent Stack UI  

For other installation options, see the [Quickstart guide](https://agentstack.beeai.dev/stable/introduction/quickstart).

---

## 2. Start the Agent Stack

Restart Agent Stack using your custom configuration file:

```bash
agentstack platform stop
agentstack platform start --set phoenix.enabled=true --set docling.enabled=true
```

Then, set up the LLM provider via:

```bash
agentstack model setup
```

---

## 3. Create a Managed MilvusDB Instance

1. Visit [https://milvus.io/](https://milvus.io/)  
2. Click **Try Managed Milvus**  
3. Create a new project  
4. Create a new cluster  
5. Save the **Public Endpoint** (`MILVUS_DB_URI`) and **Token** (`MILVUS_DB_TOKEN`)  
6. Run the following command to set the environment variables for the agent:

```bash
agentstack env add 'RAG Milvus' \
  MILVUS_DB_URI="..." \
  MILVUS_DB_TOKEN="..."
```

---

## 4. Install the Project

Clone the repository and install the project:

```bash
git clone https://github.com/jenna-winkler/rag.git
cd rag
uv sync
```

---

## 5. Run the Agent

To run the agent, first start the server:

```bash
uv run server
```

Then, interact with the `RAG Milvus` agent using the UI:

```bash
agentstack ui
```

Open your browser and navigate to [http://localhost:8334/](http://localhost:8334/).

---

## 6. Observability

Traces are stored in the Arize Phoenix instance managed by Agent Stack itself.  
Open [http://localhost:6006](http://localhost:6006) in your browser and navigate to the default project to explore the collected traces.