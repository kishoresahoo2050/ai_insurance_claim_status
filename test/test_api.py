"""
Test Suite – API Layer & Data Pipeline
Tests: claim data fetching, query routing, data preprocessing,
       anonymization, synthetic data generation, and schema validation.
Run: pytest tests/test_api.py -v
"""

import sys
import json
import csv
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def claims_json():
    """Load the full claims JSON dataset."""
    path = ROOT / "data" / "synthetic" / "claims.json"
    assert path.exists(), "Run generate_claims.py first"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def claims_dict(claims_json):
    """Claims indexed by claim_id."""
    return {c["claim_id"]: c for c in claims_json}


@pytest.fixture(scope="module")
def first_claim_id(claims_dict):
    return list(claims_dict.keys())[0]


@pytest.fixture(scope="module")
def first_claim(claims_dict, first_claim_id):
    return claims_dict[first_claim_id]


# ── 1. Claim Data File Tests ──────────────────────────────────────────────────

class TestClaimDataFiles:
    """Validate the output files from the data generation pipeline."""

    def test_json_file_exists(self):
        assert (ROOT / "data" / "synthetic" / "claims.json").exists()

    def test_csv_file_exists(self):
        assert (ROOT / "data" / "synthetic" / "claims.csv").exists()

    def test_json_is_valid(self, claims_json):
        assert isinstance(claims_json, list)
        assert len(claims_json) > 0

    def test_json_record_count(self, claims_json):
        assert len(claims_json) == 100

    def test_csv_row_count(self):
        csv_file = ROOT / "data" / "synthetic" / "claims.csv"
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 100

    def test_csv_headers_match_json(self, claims_json):
        csv_file = ROOT / "data" / "synthetic" / "claims.csv"
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            csv_headers = set(reader.fieldnames)
        json_keys = set(claims_json[0].keys())
        assert json_keys == csv_headers, "CSV headers don't match JSON keys"

    def test_json_and_csv_same_record_count(self, claims_json):
        csv_file = ROOT / "data" / "synthetic" / "claims.csv"
        with open(csv_file) as f:
            csv_count = sum(1 for _ in csv.DictReader(f))
        assert len(claims_json) == csv_count


# ── 2. Claim Schema Validation ────────────────────────────────────────────────

class TestClaimSchema:
    """Validate field types and constraints in each claim record."""

    REQUIRED_FIELDS = {
        "claim_id": str,
        "policy_number": str,
        "claim_type": str,
        "status": str,
        "status_detail": str,
        "submitted_date": str,
        "last_updated": str,
        "amount_claimed": (int, float),
        "amount_approved": (int, float),
        "claimant_name": str,
        "contact_email": str,
        "assigned_adjuster": str,
        "adjuster_phone": str,
        "documents_required": list,
        "estimated_resolution_days": int,
        "notes": str,
    }

    def test_all_required_fields_present(self, claims_json):
        for claim in claims_json:
            for field in self.REQUIRED_FIELDS:
                assert field in claim, f"Missing field '{field}' in {claim.get('claim_id')}"

    def test_field_types_correct(self, claims_json):
        for claim in claims_json[:10]:
            for field, expected_type in self.REQUIRED_FIELDS.items():
                value = claim[field]
                if value is not None:
                    assert isinstance(value, expected_type), \
                        f"Field '{field}' in {claim['claim_id']}: expected {expected_type}, got {type(value)}"

    def test_claim_id_format(self, claims_json):
        valid_prefixes = ("CLM-", "INS-", "POL-")
        for claim in claims_json:
            cid = claim["claim_id"]
            assert any(cid.startswith(p) for p in valid_prefixes), \
                f"Invalid Claim ID format: {cid}"
            suffix = cid.split("-")[1]
            assert suffix.isdigit() and len(suffix) == 6, \
                f"Claim ID suffix not 6 digits: {cid}"

    def test_policy_number_format(self, claims_json):
        for claim in claims_json:
            pn = claim["policy_number"]
            assert pn.startswith("POL-"), f"Invalid policy number: {pn}"

    def test_email_contains_at(self, claims_json):
        for claim in claims_json:
            assert "@" in claim["contact_email"], \
                f"Invalid email in {claim['claim_id']}: {claim['contact_email']}"

    def test_dates_are_strings(self, claims_json):
        for claim in claims_json:
            assert isinstance(claim["submitted_date"], str)
            assert isinstance(claim["last_updated"], str)

    def test_date_format_yyyy_mm_dd(self, claims_json):
        import re
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for claim in claims_json[:20]:
            assert pattern.match(claim["submitted_date"]), \
                f"Bad submitted_date format: {claim['submitted_date']}"
            assert pattern.match(claim["last_updated"]), \
                f"Bad last_updated format: {claim['last_updated']}"

    def test_amounts_are_positive(self, claims_json):
        for claim in claims_json:
            assert claim["amount_claimed"] > 0, \
                f"amount_claimed is zero or negative in {claim['claim_id']}"
            assert claim["amount_approved"] >= 0

    def test_resolution_days_non_negative(self, claims_json):
        for claim in claims_json:
            assert claim["estimated_resolution_days"] >= 0


# ── 3. Business Rule Validation ───────────────────────────────────────────────

class TestBusinessRules:
    """Validate business logic embedded in the synthetic data."""

    def test_approved_amount_lte_claimed(self, claims_json):
        for claim in claims_json:
            assert claim["amount_approved"] <= claim["amount_claimed"] + 0.01, \
                f"Approved > claimed in {claim['claim_id']}"

    def test_closed_and_settled_have_zero_resolution_days(self, claims_json):
        terminal = {"Approved", "Settled", "Rejected", "Closed"}
        for claim in claims_json:
            if claim["status"] in terminal:
                assert claim["estimated_resolution_days"] == 0, \
                    f"Terminal claim {claim['claim_id']} has non-zero resolution days"

    def test_status_detail_matches_status(self, claims_json):
        status_keywords = {
            "Submitted": "received",
            "Under Review": "adjuster",
            "Approved": "approved",
            "Rejected": "denied" or "could not",
            "Settled": "settled",
        }
        for claim in claims_json:
            status = claim["status"]
            if status in status_keywords:
                keyword = status_keywords[status]
                detail_lower = claim["status_detail"].lower()
                assert any(kw in detail_lower for kw in [keyword, "claim"]), \
                    f"status_detail mismatch for status '{status}' in {claim['claim_id']}"

    def test_rejection_reason_only_on_rejected(self, claims_json):
        for claim in claims_json:
            if claim["status"] != "Rejected":
                assert claim.get("rejection_reason") is None, \
                    f"Non-rejected claim {claim['claim_id']} has a rejection_reason"

    def test_all_claim_types_represented(self, claims_json):
        types_present = {c["claim_type"] for c in claims_json}
        expected = {"Auto", "Health", "Home", "Life", "Travel"}
        assert expected == types_present, \
            f"Missing claim types: {expected - types_present}"

    def test_all_statuses_represented(self, claims_json):
        statuses_present = {c["status"] for c in claims_json}
        expected = {
            "Submitted", "Under Review", "Documentation Required",
            "Investigation In Progress", "Approved", "Partially Approved",
            "Rejected", "Settled", "Appealed", "Closed"
        }
        missing = expected - statuses_present
        assert not missing, f"Missing statuses in dataset: {missing}"


# ── 4. Data Generation Script Tests ──────────────────────────────────────────

class TestDataGenerationScript:
    """Test the generate_claims.py utility functions."""

    def test_generate_claim_id_format(self):
        import sys
        sys.path.insert(0, str(ROOT / "data" / "synthetic"))
        from generate_claims import generate_claim_id
        for _ in range(20):
            cid = generate_claim_id()
            assert "-" in cid
            prefix, suffix = cid.split("-")
            assert prefix in ("CLM", "INS", "POL")
            assert len(suffix) == 6
            assert suffix.isdigit()

    def test_generate_single_claim_structure(self):
        from data.synthetic.generate_claims import generate_claim
        claim = generate_claim()
        required = ["claim_id", "policy_number", "claim_type", "status",
                    "amount_claimed", "amount_approved", "claimant_name"]
        for field in required:
            assert field in claim

    def test_generate_dataset_returns_list(self, tmp_path):
        import os
        from data.synthetic.generate_claims import generate_dataset
        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            claims = generate_dataset(10)
            assert isinstance(claims, list)
            assert len(claims) == 10
        finally:
            os.chdir(original_dir)

    def test_generated_claims_have_unique_ids(self, tmp_path):
        import os
        from data.synthetic.generate_claims import generate_dataset
        os.chdir(tmp_path)
        try:
            claims = generate_dataset(50)
            ids = [c["claim_id"] for c in claims]
            assert len(ids) == len(set(ids)), "Duplicate IDs in generated dataset"
        finally:
            os.chdir(ROOT)

    def test_random_date_within_range(self):
        from data.synthetic.generate_claims import random_date
        from datetime import datetime
        date_str = random_date(start_days_ago=30, end_days_ago=0)
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
        now = datetime.now()
        delta = (now - parsed).days
        assert 0 <= delta <= 31


# ── 5. Claim Lookup Simulation (API Layer) ────────────────────────────────────

class TestClaimLookupAPI:
    """Simulate the API layer's claim fetch behaviour."""

    def fetch_claim(self, claims_dict, claim_id):
        """Simulates claim_api.py fetch logic."""
        return claims_dict.get(claim_id)

    def test_fetch_existing_claim(self, claims_dict, first_claim_id):
        result = self.fetch_claim(claims_dict, first_claim_id)
        assert result is not None
        assert result["claim_id"] == first_claim_id

    def test_fetch_nonexistent_claim_returns_none(self, claims_dict):
        result = self.fetch_claim(claims_dict, "DOES-NOT-EXIST")
        assert result is None

    def test_fetch_claim_returns_all_fields(self, claims_dict, first_claim_id, first_claim):
        result = self.fetch_claim(claims_dict, first_claim_id)
        assert set(result.keys()) == set(first_claim.keys())

    def test_fetch_multiple_claims(self, claims_dict):
        ids = list(claims_dict.keys())[:5]
        results = [self.fetch_claim(claims_dict, cid) for cid in ids]
        assert all(r is not None for r in results)
        fetched_ids = [r["claim_id"] for r in results]
        assert fetched_ids == ids


# ── 6. Query Handler Simulation ───────────────────────────────────────────────

class TestQueryHandler:
    """Simulate the query_handler.py routing logic."""

    def classify_query(self, text):
        """Simple rule-based classifier simulating query_handler.py."""
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["status", "update", "stage"]):
            return "claim_status"
        elif any(kw in text_lower for kw in ["document", "upload", "submit", "file"]):
            return "document_info"
        elif any(kw in text_lower for kw in ["pay", "payout", "amount", "money", "transfer"]):
            return "payment_info"
        elif any(kw in text_lower for kw in ["appeal", "dispute", "reject", "denied"]):
            return "appeal_info"
        elif any(kw in text_lower for kw in ["deductible", "premium", "coverage", "policy"]):
            return "claim_explanation"
        else:
            return "general_faq"

    def test_status_query_routing(self):
        assert self.classify_query("What is the status of my claim?") == "claim_status"

    def test_document_query_routing(self):
        assert self.classify_query("What documents do I need to upload?") == "document_info"

    def test_payment_query_routing(self):
        assert self.classify_query("When will I receive my payout?") == "payment_info"

    def test_appeal_query_routing(self):
        assert self.classify_query("My claim was rejected, how do I appeal?") == "appeal_info"

    def test_explanation_query_routing(self):
        assert self.classify_query("What is a deductible?") == "claim_explanation"

    def test_general_faq_fallback(self):
        assert self.classify_query("Hello there") == "general_faq"

    def test_case_insensitive_routing(self):
        assert self.classify_query("WHAT IS THE STATUS?") == "claim_status"
        assert self.classify_query("when will i get my PAYOUT?") == "payment_info"


# ── 7. RAGAS Evaluation Data Tests ───────────────────────────────────────────

class TestRAGASEvalData:
    """Validate the RAGAS evaluation dataset structure."""

    def test_eval_questions_importable(self):
        from src.agent.ragas_eval import EVAL_QUESTIONS
        assert isinstance(EVAL_QUESTIONS, list)

    def test_eval_questions_count(self):
        from src.agent.ragas_eval import EVAL_QUESTIONS
        assert len(EVAL_QUESTIONS) >= 5

    def test_eval_question_structure(self):
        from src.agent.ragas_eval import EVAL_QUESTIONS
        for q in EVAL_QUESTIONS:
            assert "question" in q, "Missing 'question' key"
            assert "ground_truth" in q, "Missing 'ground_truth' key"
            assert "contexts" in q, "Missing 'contexts' key"
            assert isinstance(q["contexts"], list)
            assert len(q["contexts"]) > 0

    def test_eval_questions_non_empty(self):
        from src.agent.ragas_eval import EVAL_QUESTIONS
        for q in EVAL_QUESTIONS:
            assert len(q["question"]) > 10
            assert len(q["ground_truth"]) > 10
            for ctx in q["contexts"]:
                assert len(ctx) > 10

    def test_ragas_results_saved_if_exists(self):
        results_path = ROOT / "data" / "processed" / "ragas_results.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
            assert "scores" in data
            scores = data["scores"]
            for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
                assert metric in scores
                assert 0.0 <= scores[metric] <= 1.0


# ── 8. Config & Settings Tests ────────────────────────────────────────────────

class TestConfig:
    """Validate configuration files."""

    def test_model_config_exists(self):
        assert (ROOT / "config" / "model_config.json").exists()

    def test_model_config_valid_json(self):
        with open(ROOT / "config" / "model_config.json") as f:
            config = json.load(f)
        assert isinstance(config, dict)

    def test_requirements_file_exists(self):
        assert (ROOT / "requirements.txt").exists()

    def test_requirements_has_key_packages(self):
        with open(ROOT / "requirements.txt") as f:
            content = f.read().lower()
        for pkg in ["langchain", "langgraph", "streamlit", "ragas", "faker"]:
            assert pkg in content, f"Package '{pkg}' missing from requirements.txt"

    def test_run_script_exists(self):
        assert (ROOT / "run.sh").exists()

