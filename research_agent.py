from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

class ResearchState(TypedDict):
    question: str
    search_queries: list[str]
    current_query: int
    documents: list[Document]
    context: str
    answer: str
    attempts: int

class SearchPlan(BaseModel):
    reasoning: str = Field(description="Why these searches were chosen")
    queries: list[str] = Field(description="Search queries")

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
structured_llm = llm.with_structured_output(SearchPlan)
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research planner.
        
        Your job is NOT to answer questions.
        
        Your only job is to create a small number of search queries that will
        retrieve enough information to answer the user's question.
        
        Generate between 3 and 5 complementary search queries.
        Avoid redundant queries."""
    ),
    ("human", "{question}")
])

planner = planner_prompt | llm.with_structured_output(SearchPlan)

def plan(state: ResearchState):
    print("\n" + "=" * 70)
    print("PLAN")
    print("=" * 70)
    print(f"Question: {state['question']}")

    result = planner.invoke({"question": state["question"]})

    print("\nPlanner reasoning:")
    print(result.reasoning)
    print("\nSearch queries:")
    for i, query in enumerate(result.queries, 1):
        print(f"  {i}. {query}")

    return {
        "search_queries": result.queries,
        "current_query": 0,
        "documents": [],
    }

def retrieve(state: ResearchState):
    print("\n" + "=" * 70)
    print("RETRIEVE")
    print("=" * 70)
    query = state["search_queries"][state["current_query"]]
    print(f"Search query: {query}")
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents")

    existing = {
        (d.metadata["source"], d.metadata["page"])
        for d in state["documents"]
    }

    new_docs = []
    for doc in docs:
        key = (doc.metadata["source"], doc.metadata["page"])
        if key not in existing:
            new_docs.append(doc)
            existing.add(key)

            print(
                f"+ {doc.metadata.get('source')} "
                f"(page {doc.metadata.get('page')})"
            )
    print(f"Added {len(new_docs)} new documents")
    print(f"Total documents: {len(state['documents']) + len(new_docs)}")
    return {"documents": state["documents"] + new_docs}

def advance(state):
    next_query = state["current_query"] + 1
    print("\n" + "=" * 70)
    print("ADVANCE")
    print("=" * 70)
    print(
        f"Moving from query "
        f"{state['current_query'] + 1}"
        f" -> "
        f"{next_query + 1}"
    )
    return {"current_query": next_query}

def build_context(state):
    print("\n" + "=" * 70)
    print("BUILD CONTEXT")
    print("=" * 70)
    print(f"Formatting {len(state['documents'])} documents")
    context = "\n\n---\n\n".join(
        f"""Source: {doc.metadata.get("source")}
        Page: {doc.metadata.get("page")}
        
        {doc.page_content}"""
        for doc in state["documents"]
    )
    print(f"Context length: {len(context):,} characters")
    return {"context": context}


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research assistant.
        
        Answer using ONLY the provided context.
        
        If the answer is not contained in the context,
        say that you don't have enough information.
        """
            ),
            (
                "human",
                """Question:
        {question}
        
        Context:
        {context}
        """
    )
])

answer_chain = answer_prompt | llm

def answer(state):

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print("Generating final response...")

    response = answer_chain.invoke({
        "question": state["question"],
        "context": state["context"],
    })
    print("Done.")
    return {"answer": response.content}

def should_continue(state):
    print("\n" + "=" * 70)
    print("ROUTER")
    print("=" * 70)
    print(
        f"Current query: "
        f"{state['current_query'] + 1}"
        f"/"
        f"{len(state['search_queries'])}"
    )
    if state["current_query"] < len(state["search_queries"]) - 1:
        print("Decision: retrieve next query")
        return "retrieve"
    print("Decision: build final context")
    return "build_context"

builder = StateGraph(ResearchState)
builder.add_node("plan", plan)
builder.add_node("retrieve", retrieve)
builder.add_node("advance", advance)
builder.add_node("build_context", build_context)
builder.add_node("answer", answer)

builder.add_edge(START, "plan")
builder.add_edge("plan", "retrieve")
builder.add_edge("retrieve", "advance")

builder.add_conditional_edges(
    "advance",
    should_continue,
    {
        "retrieve": "retrieve",
        "build_context": "build_context",
    },
)
builder.add_edge("build_context", "answer")
builder.add_edge("answer", END)
graph = builder.compile()
#          START
#            ▼
#          plan
#            ▼
#    ┌────► retrieve
#    │       ▼
#    │     advance (are there more queries)
#    └───────┘  ╲
#              build_context
#                 ▼
#              answer
#                 ▼
#                END


# Run it

state = {
    "question": "How is JSON useful when designing WEB API's?",
    "search_queries": [],
    "documents": [],
    "answer": "",
    "attempts": 0,
}

result = graph.invoke(state)
print(result["answer"])