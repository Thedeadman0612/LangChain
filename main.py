import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.config import P
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing all the components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:

    {context}

    question: {question}

    provide the detailed answer: """
)


def format_doc(docs):
    """format retrieved  documents into a single string."""

    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)

    context = format_doc(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)

    return response.content


def retrieval_chain_with_lcel():

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_doc
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving......")

    query = "What is pinecone in machine learning?"

    # ============================================
    # Option 0: Raw invocation without RAG
    # ============================================

    print("\n" + "=" * 70)
    print("Implementation 0: Raw LLM invication (No RAG)")
    print("\n" + "=" * 70)

    result_raw = llm.invoke([HumanMessage(content=query)])

    print("\n Answer:")
    print(result_raw.content)

    # ============================================
    # Option 1: Implementation without LCEL
    # ============================================

    print("\n" + "=" * 70)
    print("Implementation 1: Implementation without LCEL")
    print("\n" + "=" * 70)

    result_without_lcel = retrieval_chain_without_lcel(query)

    print("\n Answer:")
    print(result_without_lcel)

    # ============================================
    # Option 2: Implementation with LCEL
    # ============================================

    print("\n" + "=" * 70)
    print("Implementation 2: Implementation with LCEL")
    print("\n" + "=" * 70)

    chain_with_lcel = retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})

    print("\n Answer:")
    print(result_with_lcel)
