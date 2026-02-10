import asyncio
import os
from textwrap import dedent
from typing import Annotated, cast
from a2a.types import AgentSkill, Message, Role, FilePart, FileWithUri, Part
from agentstack_sdk.platform import File
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.util.file import PlatformFileUrl
from beeai_framework.adapters.a2a.agents._utils import (
    convert_a2a_to_framework_message,
    convert_to_a2a_message,
)
from beeai_framework.adapters.agentstack.backend.chat import AgentStackChatModel
from beeai_framework.adapters.agentstack.backend.embedding import (
    AgentstackEmbeddingModel,
)
from beeai_framework.adapters.agentstack.backend.vector_store import NativeVectorStore
from beeai_framework.adapters.agentstack.serve.server import (
    AgentStackMemoryManager,
    AgentStackServer,
)
from beeai_framework.adapters.agentstack.serve.types import BaseAgentStackExtensions
from beeai_framework.adapters.langchain.backend.vector_store import LangChainVectorStore
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.events import (
    RequirementAgentFinalAnswerEvent,
    RequirementAgentSuccessEvent,
)
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.backend import ChatModelParameters, AssistantMessage
from beeai_framework.backend.text_splitter import TextSplitter
from beeai_framework.backend.types import Document
from beeai_framework.backend.vector_store import VectorStore
from beeai_framework.context import RunMiddlewareProtocol
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import (
    GlobalTrajectoryMiddleware,
    GlobalTrajectoryMiddlewareStartEvent,
    GlobalTrajectoryMiddlewareErrorEvent,
    GlobalTrajectoryMiddlewareSuccessEvent,
)
from beeai_framework.tools import Tool, ToolOutput, AnyTool
from beeai_framework.tools.search.retrieval import VectorStoreSearchTool
from agentstack_sdk.a2a.extensions import (
    EmbeddingServiceExtensionServer,
    EmbeddingServiceExtensionSpec,
    AgentDetail,
    EnvVar,
    LLMServiceExtensionServer,
    LLMServiceExtensionSpec,
    LLMServiceExtensionParams,
    TrajectoryExtensionServer,
    TrajectoryExtensionSpec,
    PlatformApiExtensionSpec,
    PlatformApiExtensionServer,
    LLMFulfillment,
    ErrorExtensionServer,
    ErrorExtensionSpec,
)

from beeai_framework.utils.strings import to_json
from langchain_milvus import Milvus
from openinference.instrumentation.beeai import BeeAIInstrumentor

BeeAIInstrumentor().instrument()

server = Server()


def create_tool_calls_trajectory_middleware(
    context: RunContext, trajectory: TrajectoryExtensionServer
) -> RunMiddlewareProtocol:
    tool_calls_trajectory_middleware = GlobalTrajectoryMiddleware(
        included=[Tool],
        excluded=[FinalAnswerTool],
        target=False,
        match_nested=True,
    )

    @tool_calls_trajectory_middleware.emitter.on("start")
    async def send_tool_call_start(
        data: GlobalTrajectoryMiddlewareStartEvent, _: EventMeta
    ) -> None:
        tool_start_event, tool_start_meta = data.origin
        tool = cast(AnyTool, tool_start_meta.creator.instance)
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'--> ' * (data.level.relative - 1)}{tool.name} (request)",
                content=to_json(tool_start_event.input, sort_keys=False, indent=4),
            )
        )

    @tool_calls_trajectory_middleware.emitter.on("success")
    async def send_tool_call_success(
        data: GlobalTrajectoryMiddlewareSuccessEvent, _: EventMeta
    ) -> None:
        tool_success_event, tool_success_meta = data.origin
        tool_output = cast(ToolOutput, tool_success_event.output)
        tool = cast(AnyTool, tool_success_meta.creator.instance)  # type: ignore[attr-defined]
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'<-- ' * (data.level.relative - 1)}{tool.name} (response)",
                content=tool_output.get_text_content(),
            )
        )

    @tool_calls_trajectory_middleware.emitter.on("error")
    async def send_tool_call_error(
        data: GlobalTrajectoryMiddlewareErrorEvent, _: EventMeta
    ) -> None:
        tool_error_event, tool_error_meta = data.origin
        tool = cast(AnyTool, tool_error_meta.creator.instance)  # type: ignore[attr-defined]
        await context.yield_async(
            trajectory.trajectory_metadata(
                title=f"{'<-- ' * (data.level.relative - 1)}{tool.name} (error)",
                content=tool_error_event.explain(),
            )
        )

    return tool_calls_trajectory_middleware


@server.agent(
    name="RAG Milvus",
    documentation_url="https://github.com/jenna-winkler/rag",
    version="0.0.1",
    default_input_modes=[
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
        "text/markdown",  # Markdown
        "text/asciidoc",  # AsciiDoc
        "text/html",  # HTML
        "application/xhtml+xml",  # XHTML
        "text/csv",  # CSV
        "image/png",  # PNG
        "image/jpeg",  # JPEG
        "image/tiff",  # TIFF
        "image/bmp",  # BMP
        "image/webp",  # WEBP
    ],
    default_output_modes=["text/plain"],
    detail=AgentDetail(
        interaction_mode="multi-turn",
        user_greeting="What would you like to read?",
        tools=[],
        framework="BeeAI",
        variables=[
            EnvVar(
                name="MILVUS_DB_URI",
                description="Full URI of your MilvusDB instance",
                required=True,
            ),
            EnvVar(
                name="MILVUS_DB_TOKEN",
                description="Token for your MilvusDB instance",
                required=True,
            ),
        ],
    ),
    skills=[
        AgentSkill(
            id="rag",
            name="RAG Agent",
            description="A Retrieval-Augmented Generation (RAG) agent that retrieves and generates text based on user queries.",
            tags=["RAG", "retrieval", "generation"],
        )
    ],
)
async def rag_agent(
    message: Message,
    context: RunContext,
    llm_ext: Annotated[
        LLMServiceExtensionServer,
        LLMServiceExtensionSpec.single_demand(suggested=("openai:gpt-5-mini",)),
    ],
    embedding_ext: Annotated[
        EmbeddingServiceExtensionServer,
        EmbeddingServiceExtensionSpec.single_demand(
            suggested=("openai:text-embedding-3-small",)
        ),
    ],
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    _: Annotated[PlatformApiExtensionServer, PlatformApiExtensionSpec()],
):
    # Init Chat Model
    llm = AgentStackChatModel(parameters=ChatModelParameters(stream=True))
    llm.set_context(llm_ext)

    # Init Embedding Model
    embedding_model = AgentstackEmbeddingModel()
    embedding_model.set_context(embedding_ext)

    # Init Vector Store
    vector_store = VectorStore.from_name(
        "langchain:Milvus",
        embedding_model=embedding_model,
        collection_name=f"beeai_ctx_{context.context_id.replace('-', '_')}",
        connection_args={
            "uri": os.environ["MILVUS_DB_URI"],
            "token": os.environ["MILVUS_DB_TOKEN"],
        },
        auto_id=True,
        # index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    assert isinstance(vector_store, LangChainVectorStore)
    assert isinstance(vector_store.vector_store, Milvus)

    # Store current message to the conversation history
    await context.store(message)

    # Initiate memory
    memory = UnconstrainedMemory()

    # Process conversation history
    all_files: list[str] = []
    new_files: list[File] = []
    async for msg in context.load_history():
        parts: list[Part] = []
        for part in msg.parts:
            if not isinstance(part.root, FilePart) or not isinstance(
                part.root.file, FileWithUri
            ):
                parts.append(part)
                continue

            file = await File.get(PlatformFileUrl(part.root.file.uri).file_id)
            all_files.append(file.filename)

            exists = bool(
                await vector_store.vector_store.asearch_by_metadata(
                    f"source_id == '{file.id}'"
                )
            )
            if not exists:
                new_files.append(file)

        await memory.add(
            convert_a2a_to_framework_message(msg.model_copy(update={"parts": parts}))
        )

    # Process new files
    for file in new_files:
        print(f"Extracting content of file {file.filename} ({file.id}).")
        extraction = await file.create_extraction()
        while extraction.status in {"pending", "in_progress"}:
            await asyncio.sleep(1)
            extraction = await file.get_extraction()
        if extraction.status != "completed":
            yield f"Failed to extract content of ${file.filename}. Ensure that the Docling is enabled."

        async with file.load_text_content() as loaded_file:
            print(f"Processing text of the {file.filename} ({file.id})")
            text_splitter = TextSplitter.from_name("langchain:MarkdownTextSplitter")
            chunks = await text_splitter.split_text(loaded_file.text)
            documents = [
                Document(
                    content=chunk,
                    metadata={"source": file.filename, "source_id": file.id},
                )
                for idx, chunk in enumerate(chunks)
            ]
            print(f"Inserting {file.filename} ({file.id}) to the Vector Store.")
            await vector_store.add_documents(documents)

    search_tool = VectorStoreSearchTool(vector_store)
    search_tool.name = "search_knowledge_base"
    search_tool.description = "Use to retrieve data from user uploaded files."

    agent = RequirementAgent(
        llm=llm,
        tools=[search_tool],
        memory=memory,
        save_intermediate_steps=False,
        instructions=f"Answer user questions accurately using the '{search_tool.name}' tool which contains all relevant documents and uploaded files.\n"
        f"List of stored in the knowledge base in the order they were processed:\n"
        + "\n".join([f"- {file}" for file in all_files]),
    )

    @agent.emitter.on("final_answer")
    async def on_final_answer(
        data: RequirementAgentFinalAnswerEvent, _: EventMeta
    ) -> None:
        # Stream the final answer chunks
        await context.yield_async(data.delta)

    tool_calls_trajectory = create_tool_calls_trajectory_middleware(context, trajectory)

    response = await agent.run(
        [], max_iterations=20, total_max_retries=10, max_retries_per_step=3
    ).middleware(tool_calls_trajectory)
    await context.store(convert_to_a2a_message(response.last_message))


def serve():
    try:
        server.run(
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", 10004)),
            configure_telemetry=True,
            context_store=PlatformContextStore(),
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    serve()
