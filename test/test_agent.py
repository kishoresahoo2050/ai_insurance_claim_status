"""
Test Suite – LangGraph Claim Agent
Tests: intent classification, claim lookup, RAG retrieval, response generation,
       graph routing, state management, and edge cases.
Run: pytest tests/test_agent.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_claims():
    """Load synthetic claim data for testing."""
    claims_file = ROOT / "data" / "synthetic" / "claims.json"
    assert claims_file.exists(), "Run generate_claims.py first to create test data"
    with open(claims_file) as f:
        claims = json.load(f)
    return {c["claim_id"]: c for c in claims}


@pytest.fixture(scope="module")
def sample_claim_id(sample_claims):
    """Return the first available claim ID."""
    return list(sample_claims.keys())[0]


@pytest.fixture(scope="module")
def rejected_claim_id(sample_claims):
    """Return the ID of a rejected claim for edge-case testing."""
    for cid, claim in sample_claims.items():
        if claim["status"] == "Rejected":
            return cid
    return list(sample_claims.keys())[0]


@pytest.fixture(scope="module")
def approved_claim_id(sample_claims):
    """Return the ID of an approved claim."""
    for cid, claim in sample_claims.items():
        if claim["status"] == "Approved":
            return cid
    return list(sample_claims.keys())[1]


@pytest.fixture
def base_state():
    """Return a minimal valid AgentState for node testing."""
    return {
        "user_input": "What is the status of my claim?",
        "claim_id": None,
        "intent": "",
        "claim_data": None,
        "rag_context": [],
        "conversation_history": [],
        "response": "",
        "error": None
    }


# ── 1. Data Layer Tests ───────────────────────────────────────────────────────

class TestSyntheticData:
    """Validate synthetic claim data structure and content."""

    def test_claims_file_exists(self):
        claims_file = ROOT / "data" / "synthetic" / "claims.json"
        assert claims_file.exists(), "claims.json not found — run generate_claims.py"

    def test_csv_file_exists(self):
        csv_file = ROOT / "data" / "synthetic" / "claims.csv"
        assert csv_file.exists(), "claims.csv not found"

    def test_claims_count(self, sample_claims):
        assert len(sample_claims) >= 50, "Expected at least 50 claims in dataset"

    def test_claim_required_fields(self, sample_claims):
        required_fields = [
            "claim_id", "policy_number", "claim_type", "status",
            "status_detail", "submitted_date", "last_updated",
            "amount_claimed", "amount_approved", "claimant_name",
            "assigned_adjuster", "adjuster_phone"
        ]
        for cid, claim in list(sample_claims.items())[:5]:
            for field in required_fields:
                assert field in claim, f"Missing field '{field}' in claim {cid}"

    def test_claim_types_valid(self, sample_claims):
        valid_types = {"Auto", "Health", "Home", "Life", "Travel"}
        for claim in sample_claims.values():
            assert claim["claim_type"] in valid_types

    def test_claim_statuses_valid(self, sample_claims):
        valid_statuses = {
            "Submitted", "Under Review", "Documentation Required",
            "Investigation In Progress", "Approved", "Partially Approved",
            "Rejected", "Settled", "Appealed", "Closed"
        }
        for claim in sample_claims.values():
            assert claim["status"] in valid_statuses

    def test_claim_amounts_non_negative(self, sample_claims):
        for claim in sample_claims.values():
            assert claim["amount_claimed"] >= 0
            assert claim["amount_approved"] >= 0

    def test_approved_claims_have_amount(self, sample_claims):
        for claim in sample_claims.values():
            if claim["status"] in ("Approved", "Settled"):
                assert claim["amount_approved"] > 0

    def test_rejected_claims_have_reason(self, sample_claims):
        for claim in sample_claims.values():
            if claim["status"] == "Rejected":
                assert claim.get("rejection_reason")

    def test_docs_required_only_for_doc_status(self, sample_claims):
        for claim in sample_claims.values():
            if claim["status"] == "Documentation Required":
                assert len(claim.get("documents_required", [])) > 0

    def test_claim_ids_are_unique(self, sample_claims):
        ids = list(sample_claims.keys())
        assert len(ids) == len(set(ids)), "Duplicate claim IDs found"


# ── 2. Glossary & Knowledge Base Tests ───────────────────────────────────────

class TestGlossary:
    """Validate the insurance glossary and FAQ knowledge base."""

    def test_glossary_imports(self):
        from data.glossary.insurance_terms import GLOSSARY, FAQ
        assert isinstance(GLOSSARY, dict)
        assert isinstance(FAQ, list)

    def test_glossary_has_entries(self):
        from data.glossary.insurance_terms import GLOSSARY
        assert len(GLOSSARY) >= 10

    def test_glossary_key_terms_present(self):
        from data.glossary.insurance_terms import GLOSSARY
        for term in ["claim", "deductible", "policy", "adjuster", "settlement"]:
            assert term in GLOSSARY, f"Key term '{term}' missing from glossary"

    def test_glossary_definitions_non_empty(self):
        from data.glossary.insurance_terms import GLOSSARY
        for term, definition in GLOSSARY.items():
            assert len(definition) > 10

    def test_faq_has_entries(self):
        from data.glossary.insurance_terms import FAQ
        assert len(FAQ) >= 5

    def test_faq_structure(self):
        from data.glossary.insurance_terms import FAQ
        for item in FAQ:
            assert "question" in item
            assert "answer" in item
            assert len(item["question"]) > 5
            assert len(item["answer"]) > 10


# ── 3. Claim Lookup Node ──────────────────────────────────────────────────────

class TestClaimLookupNode:
    """Test the claim_lookup node in isolation."""

    def test_valid_claim_id_returns_data(self, base_state, sample_claim_id):
        from src.agent.claim_agent import claim_lookup
        state = {**base_state, "claim_id": sample_claim_id}
        result = claim_lookup(state)
        assert result["claim_data"] is not None
        assert result["claim_data"]["claim_id"] == sample_claim_id
        assert result["error"] is None

    def test_invalid_claim_id_returns_error(self, base_state):
        from src.agent.claim_agent import claim_lookup
        state = {**base_state, "claim_id": "INVALID-999999"}
        result = claim_lookup(state)
        assert result["claim_data"] is None
        assert result["error"] is not None
        assert "INVALID-999999" in result["error"]

    def test_no_claim_id_returns_none(self, base_state):
        from src.agent.claim_agent import claim_lookup
        state = {**base_state, "claim_id": None}
        result = claim_lookup(state)
        assert result["claim_data"] is None
        assert result["error"] is None

    def test_claim_data_has_required_fields(self, base_state, sample_claim_id):
        from src.agent.claim_agent import claim_lookup
        state = {**base_state, "claim_id": sample_claim_id}
        result = claim_lookup(state)
        for field in ["claim_id", "status", "claim_type", "amount_claimed"]:
            assert field in result["claim_data"]


# ── 4. RAG Retriever Node ─────────────────────────────────────────────────────

class TestRAGRetrieverNode:
    """Test the RAG retriever node."""

    def test_retriever_returns_context(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {**base_state, "user_input": "What is a deductible?"}
        result = rag_retriever(state)
        assert isinstance(result["rag_context"], list)
        assert len(result["rag_context"]) > 0

    def test_retriever_context_is_strings(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {**base_state, "user_input": "How do I appeal a rejected claim?"}
        result = rag_retriever(state)
        for ctx in result["rag_context"]:
            assert isinstance(ctx, str)
            assert len(ctx) > 0

    def test_retriever_returns_relevant_content(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {**base_state, "user_input": "What documents do I need for health claim?"}
        result = rag_retriever(state)
        combined = " ".join(result["rag_context"]).lower()
        assert any(word in combined for word in ["claim", "document", "health", "insurance", "policy"])

    def test_retriever_handles_short_query(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {**base_state, "user_input": "hello"}
        result = rag_retriever(state)
        assert isinstance(result["rag_context"], list)


# ── 5. Graph Routing Logic ────────────────────────────────────────────────────

class TestGraphRouting:
    """Test conditional routing in the LangGraph."""

    def test_route_with_claim_id_goes_to_lookup(self, base_state):
        from src.agent.claim_agent import route_after_intent
        state = {**base_state, "claim_id": "CLM-123456"}
        assert route_after_intent(state) == "claim_lookup"

    def test_route_without_claim_id_goes_to_rag(self, base_state):
        from src.agent.claim_agent import route_after_intent
        state = {**base_state, "claim_id": None}
        assert route_after_intent(state) == "rag_retriever"

    def test_route_after_lookup_goes_to_rag(self, base_state):
        from src.agent.claim_agent import route_after_lookup
        assert route_after_lookup(base_state) == "rag_retriever"

    def test_graph_compiles_without_error(self):
        from src.agent.claim_agent import build_graph
        assert build_graph() is not None


# ── 6. Claims Database Tests ──────────────────────────────────────────────────

class TestClaimsDatabase:
    """Test the in-memory claims database loaded by the agent."""

    def test_claims_db_loads(self):
        from src.agent.claim_agent import CLAIMS_DB
        assert isinstance(CLAIMS_DB, dict)
        assert len(CLAIMS_DB) > 0

    def test_claims_db_indexed_by_id(self):
        from src.agent.claim_agent import CLAIMS_DB
        for cid, claim in list(CLAIMS_DB.items())[:5]:
            assert claim["claim_id"] == cid

    def test_get_sample_claim_ids(self):
        from src.agent.claim_agent import get_sample_claim_ids
        ids = get_sample_claim_ids(5)
        assert len(ids) == 5
        assert all(isinstance(i, str) for i in ids)


# ── 7. Response Generator Node (Mocked LLM) ──────────────────────────────────

class TestResponseGeneratorNode:
    """Test response_generator with mocked LLM to avoid API calls."""

    def test_response_generator_produces_output(self, base_state, sample_claim_id):
        from src.agent.claim_agent import claim_lookup, rag_retriever
        state = {**base_state, "claim_id": sample_claim_id}
        state = claim_lookup(state)
        state = rag_retriever(state)

        mock_response = MagicMock()
        mock_response.content = "Your claim is currently under review by our team."

        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from src.agent.claim_agent import response_generator
            result = response_generator(state)

        assert result["response"] == "Your claim is currently under review by our team."

    def test_response_generator_with_error_state(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {
            **base_state,
            "claim_id": "INVALID-000",
            "error": "No claim found with ID 'INVALID-000'."
        }
        state = rag_retriever(state)

        mock_response = MagicMock()
        mock_response.content = "I could not find that claim ID. Please double-check."

        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from src.agent.claim_agent import response_generator
            result = response_generator(state)

        assert len(result["response"]) > 0

    def test_response_generator_uses_conversation_history(self, base_state):
        from src.agent.claim_agent import rag_retriever
        state = {
            **base_state,
            "conversation_history": [
                {"role": "user", "content": "What is a deductible?"},
                {"role": "assistant", "content": "A deductible is the amount you pay first."}
            ]
        }
        state = rag_retriever(state)

        mock_response = MagicMock()
        mock_response.content = "Following up on your earlier question..."

        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            from src.agent.claim_agent import response_generator
            result = response_generator(state)

        assert len(result["response"]) > 0


# ── 8. Intent Classifier Node (Mocked LLM) ───────────────────────────────────

class TestIntentClassifierNode:
    """Test intent classification with mocked LLM responses."""

    def _run_classifier(self, user_input, mock_json, base_state):
        from src.agent.claim_agent import intent_classifier
        state = {**base_state, "user_input": user_input}
        mock_response = MagicMock()
        mock_response.content = json.dumps(mock_json)
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            return intent_classifier(state)

    def test_classifies_claim_status_intent(self, base_state):
        result = self._run_classifier(
            "What is the status of CLM-123456?",
            {"intent": "claim_status", "claim_id": "CLM-123456"},
            base_state
        )
        assert result["intent"] == "claim_status"
        assert result["claim_id"] == "CLM-123456"

    def test_classifies_general_faq_intent(self, base_state):
        result = self._run_classifier(
            "What is a deductible?",
            {"intent": "general_faq", "claim_id": None},
            base_state
        )
        assert result["intent"] == "general_faq"
        assert result["claim_id"] is None

    def test_classifies_payment_intent(self, base_state):
        result = self._run_classifier(
            "When will I receive my payout?",
            {"intent": "payment_info", "claim_id": None},
            base_state
        )
        assert result["intent"] == "payment_info"

    def test_handles_malformed_llm_json(self, base_state):
        from src.agent.claim_agent import intent_classifier
        state = {**base_state, "user_input": "Hello"}
        mock_response = MagicMock()
        mock_response.content = "not valid json"
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = intent_classifier(state)
        assert result["intent"] == "general_faq"

    def test_extracts_claim_id_from_message(self, base_state):
        result = self._run_classifier(
            "Check claim POL-539898 please",
            {"intent": "claim_status", "claim_id": "POL-539898"},
            base_state
        )
        assert result["claim_id"] == "POL-539898"


# ── 9. Full Agent Integration Tests (Mocked LLM) ─────────────────────────────

class TestAgentIntegration:
    """End-to-end agent tests using mocked LLM."""

    def _mock(self, content):
        m = MagicMock()
        m.content = content
        return m

    def test_run_agent_without_claim_id(self):
        from src.agent.claim_agent import run_agent
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                self._mock('{"intent": "general_faq", "claim_id": null}'),
                self._mock("A deductible is the amount you pay before your insurer covers the rest.")
            ]
            result = run_agent("What is a deductible?")
        assert len(result["response"]) > 0

    def test_run_agent_with_valid_claim_id(self, sample_claim_id):
        from src.agent.claim_agent import run_agent
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                self._mock(f'{{"intent": "claim_status", "claim_id": "{sample_claim_id}"}}'),
                self._mock("Your claim is under review. An adjuster has been assigned.")
            ]
            result = run_agent(f"What is the status of {sample_claim_id}?")
        assert result["claim_id"] == sample_claim_id
        assert result["claim_data"] is not None
        assert result["error"] is None

    def test_run_agent_with_invalid_claim_id(self):
        from src.agent.claim_agent import run_agent
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                self._mock('{"intent": "claim_status", "claim_id": "FAKE-000000"}'),
                self._mock("I could not find claim FAKE-000000.")
            ]
            result = run_agent("Check claim FAKE-000000")
        assert result["error"] is not None
        assert result["claim_data"] is None

    def test_run_agent_returns_all_keys(self):
        from src.agent.claim_agent import run_agent
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                self._mock('{"intent": "greeting", "claim_id": null}'),
                self._mock("Hello! I'm here to help with your insurance claims.")
            ]
            result = run_agent("Hello")
        for key in ["response", "intent", "claim_id", "claim_data", "error"]:
            assert key in result

    def test_run_agent_with_conversation_history(self):
        from src.agent.claim_agent import run_agent
        history = [
            {"role": "user", "content": "What is a deductible?"},
            {"role": "assistant", "content": "A deductible is the amount you pay first."}
        ]
        with patch("src.agent.claim_agent.llm") as mock_llm:
            mock_llm.invoke.side_effect = [
                self._mock('{"intent": "general_faq", "claim_id": null}'),
                self._mock("Glad to help with more questions.")
            ]
            result = run_agent("What about a premium?", conversation_history=history)
        assert "response" in result


# ── 10. Edge Case Tests ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions and unexpected inputs."""

    def _make_state(self, user_input="", claim_id=None):
        return {
            "user_input": user_input,
            "claim_id": claim_id,
            "intent": "",
            "claim_data": None,
            "rag_context": [],
            "conversation_history": [],
            "response": "",
            "error": None
        }

    def test_empty_user_input(self):
        from src.agent.claim_agent import rag_retriever
        result = rag_retriever(self._make_state(""))
        assert isinstance(result["rag_context"], list)

    def test_very_long_user_input(self):
        from src.agent.claim_agent import rag_retriever
        long_input = "What is my claim status? " * 100
        result = rag_retriever(self._make_state(long_input))
        assert isinstance(result["rag_context"], list)

    def test_rejected_claim_has_reason(self, rejected_claim_id):
        from src.agent.claim_agent import claim_lookup
        state = self._make_state(claim_id=rejected_claim_id)
        result = claim_lookup(state)
        if result["claim_data"] and result["claim_data"]["status"] == "Rejected":
            assert result["claim_data"].get("rejection_reason")

    def test_approved_claim_has_payout(self, approved_claim_id):
        from src.agent.claim_agent import claim_lookup
        state = self._make_state(claim_id=approved_claim_id)
        result = claim_lookup(state)
        if result["claim_data"] and result["claim_data"]["status"] == "Approved":
            assert result["claim_data"]["amount_approved"] > 0

    def test_special_characters_in_input(self):
        from src.agent.claim_agent import rag_retriever
        result = rag_retriever(self._make_state("What's my claim?! #urgent @support"))
        assert isinstance(result["rag_context"], list)

    def test_numeric_only_input(self):
        from src.agent.claim_agent import rag_retriever
        result = rag_retriever(self._make_state("123456"))
        assert isinstance(result["rag_context"], list)
