# Agentic RAG Example

This example demonstrates an agentic RAG architecture built with:

* **BeeAI Framework** – agent orchestration and reasoning
* **Docling** – document processing
* **Milvus** – vector store for semantic retrieval
* **Arize Phoenix** – observability and tracing
* **Agent Stack** – local platform runtime and deployment

The agent assumes that documents are **preprocessed and ingested before runtime**. At inference time, the agent only performs retrieval and reasoning.

---

## Prerequisites

* An LLM provider (OpenAI, Anthropic, watsonx, etc.) **or** Ollama

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

For other installation options, see the [quickstart](https://agentstack.beeai.dev/stable/introduction/quickstart).

---

## 2. Platform Configuration

Agent Stack supports platform configuration via a YAML config file.

Create a file called `agentstack-config.yaml` and start from the default values, then apply the changes below.

```yaml
# Disable built-in PostgreSQL
postgresql:
  enabled: false

# Point to external Milvus
externalDatabase:
  host: "milvus.example.com"
  port: 19530
  user: "milvus-user"
  password: "milvus-password"
  database: "agentstack"
  adminUser: "admin"
  adminPassword: "admin-password"
  ssl: false

# Enable Phoenix observability
phoenix:
  enabled: true

# Enable Docling document processing
docling:
  enabled: true
```

---

## 3. Start the Platform With Custom Config

Restart Agent Stack using your custom config file:

```bash
agentstack platform stop
agentstack platform start --config agentstack-config.yaml
```

Confirm the platform is running:

```bash
agentstack ui
```

---

## 4. Run the Agent

Install dependencies locally:

```bash
uv pip install "beeai-framework[agentstack]"
```

Run the agent:

```bash
uv run server
```

When the agent starts, it will:

* Register itself with Agent Stack
* Expose an agent card
* Use the **Milvus vector store**
* Emit traces to **Arize Phoenix**

---

## 5. Access the UI and Observability

* **Agent Stack UI**

  ```bash
  agentstack ui
  ```

* **Arize Phoenix**
  Available via the Agent Stack UI when enabled