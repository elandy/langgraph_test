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
    normalized_question: str
    search_queries: list[str]
    current_query: int
    documents: list[Document]
    context: str
    answer: str
    attempts: int
    grade_passed: bool

class SearchPlan(BaseModel):
    normalized_question: str = Field(
        description="Question with obvious spelling mistakes corrected and technical terms normalized"
    )
    reasoning: str = Field(description="Why these searches were chosen")
    queries: list[str] = Field(description="Search queries")

class RewriteResult(BaseModel):
    queries: list[str] = Field(description="Improved search queries")

llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a research planner for a retrieval system.

Your job is NOT to answer questions.

Your job is to:
1. Normalize the user's question by fixing obvious spelling mistakes, OCR noise,
   and incorrect casing.
2. Correct technical terms when the intended meaning is clear from context.
3. Generate a small set of search queries from the normalized question.

Rules:
- Preserve intent.
- Prefer canonical technical names and spellings.
- If a term is clearly a misspelling in context, fix it.
- In web/API contexts, prefer standard terms such as JSON, REST, HTTP, OpenAPI.
- Only keep the original misspelled token in a query if the term is genuinely ambiguous.
- Generate between 3 and 5 complementary, non-redundant queries.
- Use search-friendly keyword phrasing, not full natural-language questions.

Example:
- User question: How does JASON improve WEB API desing and interoprability?
- Normalized question: How does JSON improve web API design and interoperability?
- Good query: JSON API design interoperability benefits"""
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

    effective_question = result.normalized_question.strip() or state["question"]
    queries = list(dict.fromkeys(q.strip() for q in result.queries if q.strip()))

    print("\nNormalized question:")
    print(effective_question)

    print("\nPlanner reasoning:")
    print(result.reasoning)
    print("\nSearch queries:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    return {
        "normalized_question": effective_question,
        "search_queries": queries,
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

grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a relevance grader.

Decide if the document is useful to answer the question.

Be strict:
- Return YES only if the document contains information that directly helps answer
  the question or one of its core technical concepts.
- Return NO for documents that are only broadly related to APIs, web design, or
  software in general.
- Return NO if the match depends mostly on a misspelled term, a weak keyword
  overlap, or incidental mention of a similar-looking word.

Return YES or NO only."""
    ),
    ("human",
     """Question:
{question}

Document:
{document}""")
])

grade_chain = grade_prompt | llm

def grade_documents(state: ResearchState):

    print("\n" + "=" * 70)
    print("GRADE DOCUMENTS")
    print("=" * 70)

    filtered = []
    effective_question = state.get("normalized_question") or state["question"]

    for doc in state["documents"]:
        result = grade_chain.invoke({
            "question": effective_question,
            "document": doc.page_content[:2000],
        }).content.strip().lower()

        print(f"{doc.metadata.get('source')}:{doc.metadata.get('page')} -> {result}")

        if "yes" in result:
            filtered.append(doc)

    print(f"Kept {len(filtered)} / {len(state['documents'])}")

    return {
        "documents": filtered,
        "grade_passed": len(filtered) > 0
    }

def route_after_grade(state: ResearchState):
    print("\n" + "=" * 70)
    print("GRADE ROUTER")
    print("=" * 70)
    if state.get("grade_passed"):
        return "advance"
    if state.get("attempts", 0) >= 2:
        print("Max attempts reached → forcing advance")
        return "advance"
    return "rewrite_queries"

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a query rewriting engine for a retrieval system.

Your job is to improve search queries so they retrieve better documents.

Rules:
- Preserve original intent
- Fix obvious spelling, OCR, and terminology mistakes
- Make queries more specific and technical when needed
- Fix ambiguity
- Remove vague terms
- Prefer search-friendly phrasing (keywords over questions)
- Do NOT answer the question
- Output 1 to 3 improved search queries
"""),
    ("human",
     """Original question:
{question}

Previous query:
{query}

Problem:
{problem}
""")
])

rewrite_chain = rewrite_prompt | llm.with_structured_output(RewriteResult)

def rewrite_queries(state: ResearchState):
    current_query = state["search_queries"][state["current_query"]]
    effective_question = state.get("normalized_question") or state["question"]
    print("\n" + "=" * 70)
    print("REWRITE QUERIES")
    print("=" * 70)
    result = rewrite_chain.invoke({
        "question": effective_question,
        "query": current_query,
        "problem": "no relevant documents found"
    })
    queries = list(dict.fromkeys(q.strip() for q in result.queries if q.strip()))
    return {
        "search_queries": queries or [current_query],
        "current_query": 0,
        "attempts": state["attempts"] + 1
    }

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

        If the original question contains obvious spelling mistakes or malformed
        technical terms, answer the normalized question instead.

        Prefer a direct answer over a meta-commentary about spelling correction.

        If the answer is not contained in the context,
        say that you don't have enough information.
        """
            ),
            (
                "human",
                """Original question:
        {original_question}

        Normalized question:
        {normalized_question}

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

    effective_question = state.get("normalized_question") or state["question"]

    response = answer_chain.invoke({
        "original_question": state["question"],
        "normalized_question": effective_question,
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
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_queries", rewrite_queries)
builder.add_node("advance", advance)
builder.add_node("build_context", build_context)
builder.add_node("answer", answer)

builder.add_edge(START, "plan")
builder.add_edge("plan", "retrieve")
builder.add_edge("retrieve", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    route_after_grade,
    {
        "advance": "advance",
        "rewrite_queries": "rewrite_queries",
    },
)
builder.add_edge("rewrite_queries", "retrieve")

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
    "question": "How does JASON improve WEB API desing and interoprability?.",
    "normalized_question": "",
    "search_queries": [],
    "documents": [],
    "answer": "",
    "attempts": 0,
}
result = graph.invoke(state)
print(result["answer"])
