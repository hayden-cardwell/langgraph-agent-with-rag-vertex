import os
from typing import Annotated, Literal, Optional, Sequence
import warnings

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import vertexai
from vertexai import rag

from shared.gcp_rag_helpers import list_corpus_files, upload_file_by_index
from shared.lc_llm import get_lc_llm

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
load_dotenv()


class FileListResponse(BaseModel):
    """Structured response for file listing queries."""

    files: list[str] = Field(
        description="List of actual file names in the knowledge base"
    )
    message: str = Field(
        description="Response message about the files, using only the provided file names"
    )


class QuestionTypeResponse(BaseModel):
    """Structured response for question type determination."""

    question_type: Optional[Literal["corpus_overview", "specific_query"]] = None


class CorpusData(BaseModel):
    """Data for a single file in the corpus."""

    name: str
    description: str
    status: str
    gcs_uri: str
    created: str
    updated: str
    user_metadata: Optional[str] = None  # JSON string of custom metadata


class GraphState(BaseModel):
    """State with messages and file listing metadata."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    question_type: Optional[Literal["corpus_overview", "specific_query"]] = None
    corpus_data: Optional[list[CorpusData]] = None


# NODES
def upload_node(state: GraphState) -> dict:
    """
    Upload the file at index 1 from files_to_upload and import to RAG corpus.
    """
    bucket = os.getenv("GCS_BUCKET")
    corpus = os.getenv("RAG_CORPUS")
    if not bucket or not corpus:
        print("\n[ERROR] GCS_BUCKET and RAG_CORPUS env vars must be set for upload.")
        return {"messages": []}

    print(f"\n[INFO] Starting upload and RAG import for file at index 1...")
    result = upload_file_by_index(index=1, bucket=bucket, corpus_name=corpus)
    print(f"[INFO] Upload/Import result: {result}")
    return {"messages": []}


def retrieve_node(state: GraphState) -> dict:
    """
    Retrieve relevant context from Vertex AI RAG Engine for specific queries.
    """
    corpus = os.getenv("RAG_CORPUS")
    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION")
    top_k = int(os.getenv("RAG_TOP_K", "5"))
    user_message = state.messages[-1].content

    if not corpus:
        raise ValueError("RAG_CORPUS environment variable is required")
    if not project:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    if not location:
        raise ValueError("GCP_LOCATION environment variable is required")

    # Perform semantic retrieval
    try:
        vertexai.init(project=project, location=location)
        resp = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus)],
            text=user_message,
            rag_retrieval_config=rag.RagRetrievalConfig(top_k=top_k),
        )
    except Exception as e:
        print(f"\n[ERROR] RAG retrieval failed: {e}")
        raise

    # Extract and format context from response
    context_texts = []
    if hasattr(resp, "contexts"):
        contexts = (
            resp.contexts
            if not hasattr(resp.contexts, "contexts")
            else resp.contexts.contexts
        )
        context_texts = [c.text for c in contexts if hasattr(c, "text")]

    context_text = (
        "\n\n".join(context_texts)
        if context_texts
        else "No relevant context was found in the knowledge base."
    )

    return {
        "messages": [AIMessage(content=context_text, metadata={"type": "rag_context"})]
    }


def corpus_data_node(state: GraphState) -> dict:
    """
    Provides file data for corpus overview queries.
    Fetches and lists all available files in the corpus with detailed metadata.
    """
    corpus = os.getenv("RAG_CORPUS")

    files = list_corpus_files(corpus)

    # Extract metadata from each file
    corpus_data_list = []
    for f in files:
        # Extract status and convert to string
        status = "UNKNOWN"
        if hasattr(f, "file_status") and hasattr(f.file_status, "state"):
            status = str(f.file_status.state).split(".")[-1]  # Convert enum to string

        # Extract GCS URI and convert to string
        gcs_uri = "N/A"
        if f.gcs_source and f.gcs_source.uris:
            # uris is a repeated field (list), take first URI
            uris = list(f.gcs_source.uris)
            gcs_uri = uris[0] if uris else "N/A"

        corpus_data_list.append(
            CorpusData(
                name=f.display_name,
                description=f.description if f.description else "",
                status=status,
                gcs_uri=gcs_uri,
                created=str(f.create_time) if f.create_time else "N/A",
                updated=str(f.update_time) if f.update_time else "N/A",
                user_metadata=(
                    f.user_metadata
                    if hasattr(f, "user_metadata") and f.user_metadata
                    else None
                ),
            )
        )

    return {
        "corpus_data": corpus_data_list,
    }


def determine_question_type(state: GraphState) -> dict:
    """
    Node that utilizes the LLM to determine the type of question being asked.
    Determines if the question is about the entire corpus or seeking specific information.
    """
    llm = get_lc_llm()
    structured_llm = llm.with_structured_output(QuestionTypeResponse)

    response = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "Analyze the user's question and determine its type:\n"
                    "1. 'corpus_overview' - if asking about all files, listing files, or general overview of the knowledge base\n"
                    "2. 'specific_query' - if asking a specific question that needs to be answered from particular file(s)\n\n"
                    "Respond with ONLY 'corpus_overview' or 'specific_query'."
                )
            ),
            state.messages[-1],
        ]
    )

    print(f"\n[INFO] Question type determined: {response.question_type}")

    return {
        "question_type": response.question_type,
    }


def summary_node(state: GraphState) -> dict:
    """
    Node that generates the final answer based on messages and context.
    """
    llm = get_lc_llm()

    system_message = SystemMessage(
        content="You are a helpful assistant. Answer the user's question using the provided context."
    )

    prompt_messages: list[BaseMessage] = [system_message] + list(state.messages)
    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


# CONDITIONAL ROUTING
def route_by_question_type(state: GraphState) -> str:
    """
    Routes to different nodes based on question type.
    """
    if state.question_type == "corpus_overview":
        return "corpus_data_node"
    else:
        return "retrieve_node"


def _print_result_summary(result: dict) -> None:
    def _format_message(msg):
        role = getattr(msg, "type", msg.__class__.__name__).replace("Message", "")
        content = getattr(msg, "content", "")
        return f"[{role}] {content}"

    print("\n[RESULT] Messages")
    for i, message in enumerate(result.get("messages", []), 1):
        print(f"  {i}. {_format_message(message)}")

    question_type = result.get("question_type")
    if question_type:
        print(f"\n[RESULT] Question type: {question_type}")

    corpus_data = result.get("corpus_data") or []
    if corpus_data:
        print("\n[RESULT] Corpus data")
        for item in corpus_data:
            print(
                f"  - {item.name} ({item.status}) "
                f"=> {item.gcs_uri} updated {item.updated}"
            )


def main():
    # Create a simple graph: retrieve -> llm -> end
    g = StateGraph(GraphState)

    # Add nodes
    g.add_node("upload_node", upload_node)
    g.add_node("determine_question_type", determine_question_type)
    g.add_node("corpus_data_node", corpus_data_node)
    g.add_node("retrieve_node", retrieve_node)
    g.add_node("summary_node", summary_node)

    # g.add_edge(START, "upload_node")
    g.add_edge(START, "determine_question_type")
    g.add_edge("upload_node", "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        route_by_question_type,
        {
            "corpus_data_node": "corpus_data_node",
            "retrieve_node": "retrieve_node",
        },
    )
    # Both paths converge to summary_node
    g.add_edge("corpus_data_node", "summary_node")
    g.add_edge("retrieve_node", "summary_node")
    g.add_edge("summary_node", END)

    # Compile the graph
    lg_graph = g.compile()

    # Use default input
    user_input = "What are the differences between the crashes described in 20071229X02007.pdf and 20071231X02009.pdf?"
    print(f"Using message: {user_input}")

    # Run the graph
    print("\n" + "=" * 80)
    result = lg_graph.invoke({"messages": [HumanMessage(content=user_input)]})
    print("=" * 80)

    _print_result_summary(result)


if __name__ == "__main__":
    main()
