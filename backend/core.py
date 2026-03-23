from importlib import metadata
import os
from typing import Any, Dict
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeSparseVectorStore, PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize the embedding
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize the vector store
vectorStore = PineconeVectorStore(
    index_name="langchain-doc-index", embedding=embedding
)

# Initialize the chat model
model = init_chat_model(model="gpt-5.2", model_provider="openai")

@tool (response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about langchain."""

    retrieved_docs = vectorStore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(

        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw document
    return serialized, retrieved_docs



def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of the retrieved documents
    """

    # Create the agent with the retrieved tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about Langchain documentation."
        "You have acess to the tool that retrieves relevant documentation"
        "Use the tool to find the relevant information before answering questions. "
        "Always cite the resources you use in your answers."
        "If you cannot find the answer in the retrieved documentation, say so"
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build the message list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context document from ToolMessage with artifacts
    context_docs = []
    for message in response["messages"]:
        # check if this is the ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return { "answer": answer, "context": context_docs}

if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)