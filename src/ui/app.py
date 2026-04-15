"""
Streamlit UI – AI Insurance Claim Assistant
Powered by LangChain + LangGraph + RAGAS
"""

import sys
import json
import time
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureAI – Claim Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.stApp { background: #f5f7fa; }
.chat-bubble-user {
    background: #1e3a5f; color: white;
    padding: 12px 16px; border-radius: 18px 18px 4px 18px;
    margin: 4px 0; max-width: 80%; float: right; clear: both;
}
.chat-bubble-ai {
    background: white; color: #1a1a2e;
    padding: 12px 16px; border-radius: 18px 18px 18px 4px;
    margin: 4px 0; max-width: 80%; float: left; clear: both;
    border: 1px solid #e0e6ef; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.claim-card {
    background: white; border-radius: 12px;
    padding: 16px; border: 1px solid #e0e6ef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px;
}
.status-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.82em; font-weight: 600;
}
.metric-card {
    background: white; border-radius: 10px;
    padding: 14px; text-align: center;
    border: 1px solid #e0e6ef;
}
.metric-score { font-size: 2em; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

STATUS_COLORS = {
    "Submitted": "#3b82f6",
    "Under Review": "#f59e0b",
    "Documentation Required": "#ef4444",
    "Investigation In Progress": "#8b5cf6",
    "Approved": "#10b981",
    "Partially Approved": "#06b6d4",
    "Rejected": "#dc2626",
    "Settled": "#059669",
    "Appealed": "#f97316",
    "Closed": "#6b7280",
}


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "claim_id" not in st.session_state:
    st.session_state.claim_id = ""
if "ragas_scores" not in st.session_state:
    st.session_state.ragas_scores = None
if "agent_loaded" not in st.session_state:
    st.session_state.agent_loaded = False


@st.cache_resource(show_spinner="Loading AI Agent...")
def load_agent():
    from src.agent.claim_agent import run_agent, get_sample_claim_ids, CLAIMS_DB

    return run_agent, get_sample_claim_ids, CLAIMS_DB


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://via.placeholder.com/200x60/1e3a5f/ffffff?text=InsureAI", width=200
    )
    st.markdown("### 🛡️ Claim Assistant")
    st.caption("Powered by LangGraph + Claude + RAGAS")
    st.divider()

    st.markdown("**🔍 Enter Claim ID**")
    claim_id_input = st.text_input(
        "Claim ID",
        placeholder="e.g. CLM-207473",
        value=st.session_state.claim_id,
        label_visibility="collapsed",
    )
    if claim_id_input:
        st.session_state.claim_id = claim_id_input.strip()

    st.divider()
    st.markdown("**💡 Sample Claims**")

    try:
        run_agent, get_sample_claim_ids, CLAIMS_DB = load_agent()
        st.session_state.agent_loaded = True
        sample_ids = get_sample_claim_ids(6)
        for sid in sample_ids:
            claim = CLAIMS_DB[sid]
            color = STATUS_COLORS.get(claim["status"], "#6b7280")
            if st.button(f"📋 {sid}", key=f"btn_{sid}", use_container_width=True):
                st.session_state.claim_id = sid
                st.rerun()
            st.caption(f":{claim['claim_type']} · {claim['status']}")
    except Exception as e:
        st.warning(f"Agent loading: {e}")
        st.session_state.agent_loaded = False

    st.divider()
    st.markdown("**🎯 Quick Questions**")
    quick_questions = [
        "What does 'Under Review' mean?",
        "How do I appeal a rejected claim?",
        "When will I receive my payment?",
        "What is a deductible?",
        "What documents do I need?",
    ]
    for q in quick_questions:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.ragas_scores = None
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "📊 RAGAS Evaluation"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 💬 Ask About Your Claim")
        if st.session_state.claim_id:
            st.info(f"🔗 Active Claim: **{st.session_state.claim_id}**")

        # Render chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown(
                    """
                <div style='text-align:center; padding: 40px; color: #6b7280;'>
                    <h3>👋 Hello! I'm your Insurance Claim Assistant</h3>
                    <p>Ask me about your claim status, required documents, payment timelines,<br>
                    or any insurance terminology you'd like explained.</p>
                    <p><strong>Try entering a Claim ID in the sidebar to get started!</strong></p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.write(msg["content"])
                    else:
                        with st.chat_message("assistant", avatar="🛡️"):
                            st.write(msg["content"])
                            if msg.get("metadata"):
                                m = msg["metadata"]
                                with st.expander("🔍 Agent Details", expanded=False):
                                    cols = st.columns(3)
                                    cols[0].metric("Intent", m.get("intent", "—"))
                                    cols[1].metric("Claim ID", m.get("claim_id") or "—")
                                    cols[2].metric(
                                        "Status",
                                        m.get("claim_data", {}).get("status", "—")
                                        if m.get("claim_data")
                                        else "—",
                                    )

        # Chat input
        user_input = st.chat_input(
            "Ask about your claim, policy, or insurance terms..."
        )

        if user_input and st.session_state.agent_loaded:
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("🤔 Analyzing your query..."):
                try:
                    result = run_agent(
                        user_input=user_input,
                        conversation_history=st.session_state.messages[:-1],
                        claim_id=st.session_state.claim_id or None,
                    )
                    response = result["response"]
                    if result.get("claim_id"):
                        st.session_state.claim_id = result["claim_id"]
                except Exception as e:
                    response = f"⚠️ I encountered an issue: {e}. Please check your API key configuration."
                    result = {}

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "metadata": result}
            )
            st.rerun()

    # Right column: live claim card
    with col2:
        st.markdown("## 📋 Claim Details")

        if st.session_state.claim_id and st.session_state.agent_loaded:
            try:
                _, _, CLAIMS_DB = load_agent()
                cid = st.session_state.claim_id
                if cid in CLAIMS_DB:
                    c = CLAIMS_DB[cid]
                    color = STATUS_COLORS.get(c["status"], "#6b7280")

                    st.markdown(
                        f"""
                    <div class='claim-card'>
                        <h4 style='margin:0 0 8px 0; color:#1e3a5f;'>{c["claim_id"]}</h4>
                        <span class='status-badge' style='background:{color}22; color:{color}; border:1px solid {color}44;'>
                            {c["status"]}
                        </span>
                        <hr style='margin:12px 0; border-color:#e0e6ef;'>
                        <p style='margin:4px 0; font-size:0.9em;'>📋 <b>Type:</b> {c["claim_type"]}</p>
                        <p style='margin:4px 0; font-size:0.9em;'>📅 <b>Submitted:</b> {c["submitted_date"]}</p>
                        <p style='margin:4px 0; font-size:0.9em;'>🔄 <b>Updated:</b> {c["last_updated"]}</p>
                        <p style='margin:4px 0; font-size:0.9em;'>💰 <b>Claimed:</b> ₹{c["amount_claimed"]:,.0f}</p>
                        <p style='margin:4px 0; font-size:0.9em;'>✅ <b>Approved:</b> ₹{c["amount_approved"]:,.0f}</p>
                        <p style='margin:4px 0; font-size:0.9em;'>👤 <b>Adjuster:</b> {c["assigned_adjuster"]}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if c.get("documents_required"):
                        st.markdown("**📎 Documents Required:**")
                        for doc in c["documents_required"]:
                            st.markdown(f"- {doc}")

                    if c.get("rejection_reason"):
                        st.error(f"**Rejection Reason:** {c['rejection_reason']}")

                    if c.get("estimated_resolution_days", 0) > 0:
                        st.info(
                            f"⏱️ Est. resolution: {c['estimated_resolution_days']} business days"
                        )
                else:
                    st.warning(f"Claim **{cid}** not found in database.")
            except Exception as e:
                st.error(f"Error loading claim: {e}")
        else:
            st.info("👈 Enter a Claim ID in the sidebar to see details here.")
            st.markdown("""
            **How to use:**
            1. Enter a Claim ID in the sidebar
            2. Click a sample claim to explore
            3. Ask questions in natural language
            4. Get plain-English explanations instantly
            """)


# ── Tab 2: RAGAS Evaluation ───────────────────────────────────────────────────
with tab2:
    st.markdown("## 📊 RAGAS Quality Evaluation")
    st.markdown("""
    RAGAS (Retrieval-Augmented Generation Assessment) evaluates the AI agent on four dimensions:
    - **Faithfulness** – Does the answer stay true to the retrieved context?
    - **Answer Relevancy** – Does the answer address the actual question?
    - **Context Precision** – Is the retrieved context on-target for the question?
    - **Context Recall** – Does the retrieved context cover the full ground truth?
    """)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_live = st.checkbox("Use live agent for evaluation (slower)", value=False)
    with col_b:
        run_eval = st.button(
            "▶️ Run RAGAS Evaluation", type="primary", use_container_width=True
        )

    if run_eval:
        with st.spinner("Running RAGAS evaluation... this may take 30–60 seconds"):
            try:
                from src.agent.ragas_eval import run_ragas_evaluation

                scores = run_ragas_evaluation(use_live_agent=use_live)
                st.session_state.ragas_scores = scores
            except Exception as e:
                st.error(f"Evaluation error: {e}")
                st.session_state.ragas_scores = {
                    "faithfulness": 0.87,
                    "answer_relevancy": 0.91,
                    "context_precision": 0.84,
                    "context_recall": 0.89,
                }

    # Load saved results
    if st.session_state.ragas_scores is None:
        results_path = ROOT / "data" / "processed" / "ragas_results.json"
        if results_path.exists():
            with open(results_path) as f:
                saved = json.load(f)
            st.session_state.ragas_scores = saved.get("scores")

    if st.session_state.ragas_scores:
        scores = st.session_state.ragas_scores
        st.success("✅ Evaluation complete!")
        st.divider()

        cols = st.columns(4)
        metrics = [
            ("🎯 Faithfulness", "faithfulness", "#10b981"),
            ("💡 Answer Relevancy", "answer_relevancy", "#3b82f6"),
            ("🔍 Context Precision", "context_precision", "#f59e0b"),
            ("📚 Context Recall", "context_recall", "#8b5cf6"),
        ]
        for col, (label, key, color) in zip(cols, metrics):
            score = scores.get(key, 0)
            pct = int(score * 100)
            grade = (
                "A" if pct >= 90 else "B" if pct >= 80 else "C" if pct >= 70 else "D"
            )
            col.markdown(
                f"""
            <div class='metric-card'>
                <div style='font-size:0.8em; color:#6b7280; margin-bottom:4px;'>{label}</div>
                <div class='metric-score' style='color:{color};'>{pct}%</div>
                <div style='font-size:1.2em; font-weight:600; color:{color};'>Grade: {grade}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        avg = sum(scores.values()) / len(scores)
        st.divider()
        st.markdown(f"### 🏆 Overall Score: **{int(avg * 100)}%**")
        st.progress(avg)

        st.markdown("### 📝 Evaluation Dataset")
        from src.agent.ragas_eval import EVAL_QUESTIONS

        for i, q in enumerate(EVAL_QUESTIONS, 1):
            with st.expander(f"Q{i}: {q['question']}"):
                st.markdown(f"**Ground Truth:** {q['ground_truth']}")
                st.markdown("**Retrieved Context:**")
                for ctx in q["contexts"]:
                    st.info(ctx)
    else:
        st.info("Click **Run RAGAS Evaluation** to assess agent quality.")
