"""
LangGraph-powered Insurance Claim Agent
Nodes: intent_classifier → claim_lookup → response_generator → output
"""

import json
import os
import sys
from typing import TypedDict, Optional, List
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "synthetic"
GLOSSARY_DIR = ROOT / "data" / "glossary"

sys.path.insert(0, str(ROOT))
from data.glossary.insurance_terms import GLOSSARY, FAQ

# ── LLM ─────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2,
)


# ── Load claim database ──────────────────────────────────────────────────────
def load_claims() -> dict:
    claims_file = DATA_DIR / "claims.json"
    if not claims_file.exists():
        return {}
    with open(claims_file) as f:
        claims = json.load(f)
    return {c["claim_id"]: c for c in claims}


CLAIMS_DB = load_claims()


# ── Build RAG vector store from glossary + FAQs ──────────────────────────────
def build_vectorstore():
    docs = []
    for term, definition in GLOSSARY.items():
        docs.append(
            Document(
                page_content=f"Term: {term.replace('_', ' ').title()}\nDefinition: {definition}",
                metadata={"type": "glossary", "term": term},
            )
        )
    for faq in FAQ:
        docs.append(
            Document(
                page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
                metadata={"type": "faq"},
            )
        )
    embeddings = FakeEmbeddings(size=128)
    return FAISS.from_documents(docs, embeddings)


VECTORSTORE = build_vectorstore()


# ── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_input: str
    claim_id: Optional[str]
    intent: str
    claim_data: Optional[dict]
    rag_context: List[str]
    conversation_history: List[dict]
    response: str
    error: Optional[str]


# ── Node 1: Intent Classifier ────────────────────────────────────────────────
def intent_classifier(state: AgentState) -> AgentState:
    """Classify user intent and extract claim ID if present."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""You are an intent classifier for an insurance claims system.
Extract the user's intent and any claim ID from their message.

Intents:
- claim_status: User wants to know the status of a specific claim
- claim_explanation: User wants explanation of claim terminology or process  
- document_info: User asking about required documents
- payment_info: User asking about payment or payout
- appeal_info: User asking about appeals
- general_faq: General insurance question
- greeting: Hello or greeting
- unknown: Cannot determine intent

Respond ONLY as JSON:
{"intent": "<intent>", "claim_id": "<ID or null>"}
"""
            ),
            HumanMessage(content=state["user_input"]),
        ]
    )
    result = llm.invoke(prompt.format_messages())
    try:
        text = result.content.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        parsed = json.loads(text)
        state["intent"] = parsed.get("intent", "general_faq")
        state["claim_id"] = parsed.get("claim_id") or state.get("claim_id")
    except Exception:
        state["intent"] = "general_faq"
    return state


# ── Node 2: Claim Lookup ─────────────────────────────────────────────────────
def claim_lookup(state: AgentState) -> AgentState:
    """Fetch claim data if a claim ID is present."""
    claim_id = state.get("claim_id")
    if claim_id and claim_id in CLAIMS_DB:
        state["claim_data"] = CLAIMS_DB[claim_id]
        state["error"] = None
    elif claim_id:
        state["claim_data"] = None
        state["error"] = (
            f"No claim found with ID '{claim_id}'. Please check the ID and try again."
        )
    else:
        state["claim_data"] = None
        state["error"] = None
    return state


# ── Node 3: RAG Retrieval ────────────────────────────────────────────────────
def rag_retriever(state: AgentState) -> AgentState:
    """Retrieve relevant glossary/FAQ context using semantic search."""
    query = state["user_input"]
    docs = VECTORSTORE.similarity_search(query, k=3)
    state["rag_context"] = [d.page_content for d in docs]
    return state


# ── Node 4: Response Generator ───────────────────────────────────────────────
def response_generator(state: AgentState) -> AgentState:
    """Generate a clear, empathetic, plain-English response."""

    claim_section = ""
    if state.get("error"):
        claim_section = f"\n⚠️ Claim Lookup Error: {state['error']}"
    elif state.get("claim_data"):
        c = state["claim_data"]
        docs = c.get("documents_required", [])
        doc_str = "\n  • " + "\n  • ".join(docs) if docs else "None at this time"
        rej = c.get("rejection_reason")
        rej_str = f"\n• Rejection Reason: {rej}" if rej else ""
        amt_approved = (
            f"₹{c['amount_approved']:,.2f}" if c["amount_approved"] > 0 else "Pending"
        )

        claim_section = f"""
Claim Details:
• Claim ID: {c["claim_id"]}
• Policy Number: {c["policy_number"]}
• Claim Type: {c["claim_type"]}
• Status: {c["status"]}
• Submitted: {c["submitted_date"]}
• Last Updated: {c["last_updated"]}
• Amount Claimed: ₹{c["amount_claimed"]:,.2f}
• Amount Approved: {amt_approved}
• Assigned Adjuster: {c["assigned_adjuster"]} ({c["adjuster_phone"]}){rej_str}
• Documents Required: {doc_str}
• Status Explanation: {c["status_detail"]}
"""

    rag_section = "\n\n".join(state.get("rag_context", []))
    history = state.get("conversation_history", [])
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-4:]
    )

    system_prompt = """You are a compassionate, knowledgeable insurance claims assistant.
Your job is to explain claim statuses and processes in simple, friendly language.
Avoid jargon. Be specific and reassuring. If a claim is rejected, acknowledge 
the frustration and explain next steps clearly. Always end with an offer to help further.

Keep responses concise (3–5 sentences for simple queries, up to 10 for complex ones).
Use bullet points only when listing multiple items. Never make up claim data.
"""

    user_prompt = f"""Conversation History:
{history_text}

Current Question: {state["user_input"]}

{claim_section}

Relevant Knowledge Base:
{rag_section}

Please provide a clear, helpful response tailored to this customer's situation.
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    result = llm.invoke(messages)
    state["response"] = result.content
    return state


# ── Routing Logic ────────────────────────────────────────────────────────────
def route_after_intent(state: AgentState) -> str:
    if state.get("claim_id"):
        return "claim_lookup"
    return "rag_retriever"


def route_after_lookup(state: AgentState) -> str:
    return "rag_retriever"


# ── Build LangGraph ──────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("claim_lookup", claim_lookup)
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("response_generator", response_generator)

    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {"claim_lookup": "claim_lookup", "rag_retriever": "rag_retriever"},
    )

    graph.add_edge("claim_lookup", "rag_retriever")
    graph.add_edge("rag_retriever", "response_generator")
    graph.add_edge("response_generator", END)

    return graph.compile()


AGENT = build_graph()


# ── Public interface ─────────────────────────────────────────────────────────
def run_agent(
    user_input: str, conversation_history: list = None, claim_id: str = None
) -> dict:
    """Run the LangGraph agent and return response + metadata."""
    state: AgentState = {
        "user_input": user_input,
        "claim_id": claim_id,
        "intent": "",
        "claim_data": None,
        "rag_context": [],
        "conversation_history": conversation_history or [],
        "response": "",
        "error": None,
    }
    result = AGENT.invoke(state)
    return {
        "response": result["response"],
        "intent": result["intent"],
        "claim_id": result.get("claim_id"),
        "claim_data": result.get("claim_data"),
        "error": result.get("error"),
    }


# ── Available claim IDs helper ───────────────────────────────────────────────
def get_sample_claim_ids(n=5):
    return list(CLAIMS_DB.keys())[:n]


if __name__ == "__main__":
    print("Sample Claim IDs:", get_sample_claim_ids())
    cid = get_sample_claim_ids(1)[0]
    result = run_agent(f"What is the status of claim {cid}?")
    print("\nResponse:", result["response"])
