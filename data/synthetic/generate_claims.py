"""
Synthetic Insurance Claim Data Generator
Generates realistic, anonymized claim datasets for prototype use.
"""

import json
import csv
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)

CLAIM_TYPES = ["Auto", "Health", "Home", "Life", "Travel"]

STATUSES = [
    "Submitted",
    "Under Review",
    "Documentation Required",
    "Investigation In Progress",
    "Approved",
    "Partially Approved",
    "Rejected",
    "Settled",
    "Appealed",
    "Closed"
]

STATUS_DETAILS = {
    "Submitted": "Your claim has been received and is pending initial review by our claims team.",
    "Under Review": "A claims adjuster has been assigned and is currently evaluating your submission.",
    "Documentation Required": "We need additional documents to proceed. Please submit the requested items within 15 days.",
    "Investigation In Progress": "A specialist is conducting a detailed investigation. This may take 5–10 business days.",
    "Approved": "Your claim has been fully approved. Payment will be processed within 3–5 business days.",
    "Partially Approved": "Your claim was approved for a portion of the requested amount due to policy limits or deductibles.",
    "Rejected": "Your claim could not be approved based on the current policy terms. See rejection reason for details.",
    "Settled": "Your claim has been settled and payment has been issued. Check your registered account.",
    "Appealed": "Your appeal is under review. An independent adjuster will reassess within 10 business days.",
    "Closed": "This claim has been closed. Contact support if you believe this was in error."
}

DOCUMENTS_REQUIRED = {
    "Auto": ["Police report", "Repair estimate", "Photos of damage", "Driver's license copy"],
    "Health": ["Medical bills", "Doctor's notes", "Prescription receipts", "Pre-authorization form"],
    "Home": ["Property photos", "Contractor estimate", "Proof of ownership", "Weather incident report"],
    "Life": ["Death certificate", "Beneficiary ID proof", "Policy document", "Medical examiner report"],
    "Travel": ["Flight cancellation proof", "Medical receipts abroad", "Passport copy", "Trip itinerary"]
}

REJECTION_REASONS = [
    "Policy exclusion: pre-existing condition",
    "Claim filed after the permitted window",
    "Insufficient documentation provided",
    "Incident not covered under current plan",
    "Duplicate claim detected"
]


def generate_claim_id():
    prefix = random.choice(["CLM", "INS", "POL"])
    return f"{prefix}-{random.randint(100000, 999999)}"


def random_date(start_days_ago=180, end_days_ago=0):
    days = random.randint(end_days_ago, start_days_ago)
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def generate_claim():
    claim_type = random.choice(CLAIM_TYPES)
    status = random.choice(STATUSES)
    submitted_date = random_date(180, 30)
    last_updated = random_date(29, 0)
    amount_claimed = round(random.uniform(500, 150000), 2)
    amount_approved = (
        round(amount_claimed * random.uniform(0.5, 1.0), 2)
        if status in ["Approved", "Partially Approved", "Settled"]
        else 0
    )

    claim = {
        "claim_id": generate_claim_id(),
        "policy_number": f"POL-{random.randint(10000, 99999)}",
        "claim_type": claim_type,
        "status": status,
        "status_detail": STATUS_DETAILS[status],
        "submitted_date": submitted_date,
        "last_updated": last_updated,
        "amount_claimed": amount_claimed,
        "amount_approved": amount_approved,
        "claimant_name": fake.name(),
        "contact_email": fake.email(),
        "assigned_adjuster": fake.name(),
        "adjuster_phone": fake.phone_number(),
        "documents_required": (
            DOCUMENTS_REQUIRED[claim_type]
            if status == "Documentation Required"
            else []
        ),
        "rejection_reason": (
            random.choice(REJECTION_REASONS) if status == "Rejected" else None
        ),
        "estimated_resolution_days": (
            random.randint(3, 30) if status not in ["Approved", "Settled", "Rejected", "Closed"] else 0
        ),
        "notes": fake.sentence(nb_words=12)
    }
    return claim


def generate_dataset(n=100):
    claims = [generate_claim() for _ in range(n)]

    # Save as JSON
    with open("claims.json", "w") as f:
        json.dump(claims, f, indent=2)

    # Save as CSV
    with open("claims.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claims[0].keys())
        writer.writeheader()
        for c in claims:
            row = c.copy()
            row["documents_required"] = "; ".join(row["documents_required"])
            writer.writerow(row)

    print(f"Generated {n} synthetic claims → claims.json and claims.csv")
    return claims


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    generate_dataset(100)
