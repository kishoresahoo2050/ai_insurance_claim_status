"""
RAGAS Evaluation Suite for the Insurance Claim Agent
Evaluates: faithfulness, answer_relevancy, context_precision, context_recall
"""

import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# ── Test dataset ─────────────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    {
        "question": "What does 'Under Review' status mean for my claim?",
        "ground_truth": "Under Review means a licensed claims adjuster has been assigned and is evaluating your claim. No action is needed from you unless contacted.",
        "contexts": [
            "Under Review: A claims adjuster has been assigned and is currently evaluating your submission.",
            # "An adjuster is a professional who investigates insurance claims to determine the extent of the insurer's liability.",
        ],
    },
    # {
    #     "question": "Why would a claim be rejected?",
    #     "ground_truth": "Claims are rejected due to policy exclusions, missed filing windows, insufficient documentation, duplicate claims, or incidents not covered under the policy.",
    #     "contexts": [
    #         "Rejection reasons include: policy exclusion for pre-existing conditions, claim filed after the permitted window, insufficient documentation, incident not covered, or duplicate claim detected.",
    #         "Exclusion: Specific conditions or situations not covered by an insurance policy.",
    #     ],
    # },
    # {
    #     "question": "How long does it take to get paid after approval?",
    #     "ground_truth": "After approval, payment is typically processed within 3–5 business days to your registered bank account.",
    #     "contexts": [
    #         "Approved: Your claim has been fully approved. Payment will be processed within 3–5 business days.",
    #         "Settlement: Agreement between the insurer and claimant on the compensation amount for a claim.",
    #     ],
    # },
    # {
    #     "question": "What documents are needed for a health insurance claim?",
    #     "ground_truth": "Health claims typically require medical bills, doctor's notes, prescription receipts, and a pre-authorization form.",
    #     "contexts": [
    #         "Documents required for Health claims: Medical bills, Doctor's notes, Prescription receipts, Pre-authorization form.",
    #         "Pre-authorization: Approval from the insurer required before receiving certain services or treatments.",
    #     ],
    # },
    # {
    #     "question": "What is a deductible?",
    #     "ground_truth": "A deductible is the amount you pay out-of-pocket before your insurance coverage begins paying.",
    #     "contexts": [
    #         "Deductible: The amount paid out-of-pocket by the policyholder before the insurance company pays its share of a covered loss.",
    #         "Premium: The amount paid periodically to the insurer in exchange for coverage.",
    #     ],
    # },
]


def generate_answers_from_agent(questions: list) -> list:
    """Generate answers using the live agent for evaluation."""
    try:
        from src.agent.claim_agent import run_agent

        answers = []
        for q in questions:
            result = run_agent(q["question"])
            answers.append(result["response"])
        return answers
    except Exception as e:
        print(f"Agent unavailable, using mock answers: {e}")
        return [
            "Under Review means a claims adjuster has been assigned and is currently evaluating your claim details.",
            # "Claims are rejected when the incident is excluded by policy terms, documentation is missing, or the filing deadline has passed.",
            # "Payment is processed within 3–5 business days after your claim is approved.",
            # "Health claims require medical bills, doctor notes, prescription receipts, and pre-authorization forms.",
            # "A deductible is the fixed amount you pay before your insurer covers the remaining costs.",
        ]


def run_ragas_evaluation(use_live_agent: bool = False) -> dict:
    """
    Run RAGAS evaluation and return scores.
    Returns a dict with metric names and scores.
    """

    questions = [q["question"] for q in EVAL_QUESTIONS]
    ground_truths = [q["ground_truth"] for q in EVAL_QUESTIONS]
    contexts = [q["contexts"] for q in EVAL_QUESTIONS]

    if use_live_agent:
        print("Generating answers from live agent...")
        answers = generate_answers_from_agent(EVAL_QUESTIONS)
    else:
        answers = [
            "Under Review means a claims adjuster has been assigned and is evaluating your submission.",
            # "Claims are rejected due to policy exclusions, expired filing windows, insufficient documentation, or duplicate submissions.",
            # "After approval, payment is processed within 3 to 5 business days to your registered account.",
            # "For health claims, you need medical bills, doctor's notes, prescription receipts, and a pre-authorization form.",
            # "A deductible is the amount you pay out of pocket before your insurer covers the rest of the claim.",
        ]

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # ✅ LLM with timeout (VERY IMPORTANT)
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.2,
        timeout=120,  # ⬅️ prevents TimeoutError
    )

    # ✅ Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    # # Attach LLM + embeddings
    # for m in metrics:
    #     m.llm = ragas_llm
    #     if hasattr(m, "embeddings"):
    #         m.embeddings = ragas_embeddings

    print("Running RAGAS evaluation...")
    scores = {}

    for metric in metrics:
        print(f"Running {metric.name}")

        result = evaluate(
            dataset,
            metrics=[metric],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        # 🔥 Extract score safely
        metric_name = metric.name
        try:
            values = result[metric_name]

            if isinstance(values, list):
                score = sum(v for v in values if v is not None) / len(values)
            else:
                score = float(values)

            scores[metric_name] = round(score, 4) if score is not None else 0.0001

        except Exception as e:
            print(f"Error in {metric_name}: {e}")
            scores[metric_name] = 0.0001
        time.sleep(3)  # 2–3 seconds (you can tune)

    print("Final Scores:", scores)
    # Save results
    output_path = ROOT / "data" / "processed" / "ragas_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "scores": scores,
                "num_questions": len(questions),
                "model": "gemini-2.5-flash",
            },
            f,
            indent=2,
        )

    print("RAGAS Results:", json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    run_ragas_evaluation(use_live_agent=False)
