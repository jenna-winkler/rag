from typing import Annotated

from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.backend.embedding import AgentstackEmbeddingModel
from beeai_framework.adapters.agentstack.backend.vector_store import NativeVectorStore
from beeai_framework.adapters.agentstack.serve.server import (
    AgentStackMemoryManager,
    AgentStackServer,
)
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModelParameters
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool

try:
    from agentstack_sdk.a2a.extensions import (
        EmbeddingServiceExtensionServer,
        EmbeddingServiceExtensionSpec,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\n"
        "Run 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


def run() -> None:
    """
    RAG Agent that assumes:
    - Documents are preprocessed (e.g. via Docling)
    - Embeddings are already stored in Milvus
    - Vector store is configured at deploy time via Agent Stack / Helm
    """

    llm = AgentStackChatModel(
        preferred_models=[
            "openai:gpt-4o",
            "ollama:llama3.1:8b",
        ],
        parameters=ChatModelParameters(stream=True),
    )

    embedding_model = AgentstackEmbeddingModel(
        preferred_models=["ollama:nomic-embed-text:latest"]
    )

    vector_store = NativeVectorStore(embedding_model)

    agent = RequirementAgent(
        llm=llm,
        tools=[
            VectorStoreSearchTool(
                vector_store=vector_store,
            )
        ],
        memory=UnconstrainedMemory(),
    )

    class CustomExtensions(BaseAgentStackExtensions):
        """
        Exposes embedding service to Agent Stack so that:
        - Ingestion jobs
        - Other agents
        - Platform services
        can reuse the same embedding model
        """

        embedding: Annotated[
            EmbeddingServiceExtensionServer,
            EmbeddingServiceExtensionSpec.single_demand(
                suggested=tuple(embedding_model.preferred_models)
            ),
        ]

    server = AgentStackServer(
        memory_manager=AgentStackMemoryManager()
    )

    server.register(
        agent,
        name="RAG Agent",
        extensions=CustomExtensions,
    )

    server.serve()


if __name__ == "__main__":
    run()
