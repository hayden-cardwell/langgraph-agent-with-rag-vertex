import base64
import os
from pathlib import Path
from typing import Annotated, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from shared.lc_llm import get_lc_llm

load_dotenv()


class GraphState(BaseModel):
    """Simple state with just messages."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_path: str = ""


def load_pdf_node(state: GraphState) -> dict:
    """
    Load PDF file and encode it as base64 to send directly to the LLM.
    """
    pdf_path = state.pdf_path
    if not pdf_path or not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        return {"messages": []}

    print(f"[INFO] Loading and encoding PDF: {pdf_path}")

    # Read and encode the PDF as base64
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    print(f"[INFO] PDF encoded, size: {len(pdf_base64)} characters")

    # Create a message with the base64-encoded PDF
    message = HumanMessage(
        content=[
            {"type": "text", "text": state.messages[-1].content},
            {
                "type": "media",
                "mime_type": "application/pdf",
                "data": pdf_base64,
            },
        ]
    )

    return {"messages": [message]}


def llm_node(state: GraphState) -> dict:
    """
    Call the LLM with the current messages (including PDF).
    """
    llm = get_lc_llm()
    response = llm.invoke(state.messages)
    return {"messages": [response]}


def main():
    # Create a simple graph: load_pdf -> llm -> end
    g = StateGraph(GraphState)

    # Add nodes
    g.add_node("load_pdf", load_pdf_node)
    g.add_node("llm", llm_node)

    # Add edges
    g.add_edge(START, "load_pdf")
    g.add_edge("load_pdf", "llm")
    g.add_edge("llm", END)

    # Compile the graph
    lg_graph = g.compile()

    # Use the specified PDF from files_to_upload directory
    files_dir = Path(__file__).parent.parent / "files_to_upload"
    pdf_path = str(files_dir / "20071229X02007.pdf")

    user_input = "What was the Runway Length of the airport? What page is this on?"

    print(f"Using PDF: {pdf_path}")
    print(f"Question: {user_input}")

    # Run the graph
    print("\n" + "=" * 80)
    result = lg_graph.invoke(
        {"messages": [HumanMessage(content=user_input)], "pdf_path": pdf_path}
    )
    print("=" * 80)

    # Print final result
    print("\nFinal response:")
    if result["messages"]:
        print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
