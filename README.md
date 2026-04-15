# 🛡️ InsureAI – AI Insurance Claim Assistant
### AI Friday Season 2 · Problem Statement #5

> An AI-powered assistant that explains insurance claim statuses and processes in plain English — reducing support workload and empowering customers with clarity.

---

## 📌 Problem Statement

Insurance customers frequently struggle to understand the status and details of their claims due to complex terminology and convoluted processes. Customer service centres receive repetitive inquiries about claim stages, required documentation, and payout timelines — leading to long wait times and inconsistent explanations.

**InsureAI** solves this by providing a conversational AI assistant that:
- Interprets natural language questions about claims
- Fetches and explains claim data using a unique Claim ID
- Translates insurance jargon into plain, empathetic language
- Evaluates its own answer quality using RAGAS metrics

---

## 🧰 Technology Stack

| Layer | Technology |
|---|---|
| AI Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Integration | [LangChain](https://github.com/langchain-ai/langchain) + Claude (Anthropic) |
| LLM Model | `claude-haiku-4-5` (fast, efficient) |
| RAG Vector Store | FAISS + LangChain Community |
| Evaluation Framework | [RAGAS](https://github.com/explodinggradients/ragas) |
| Frontend UI | [Streamlit](https://streamlit.io/) |
| Synthetic Data | [Faker](https://faker.readthedocs.io/) |

---

## 📁 Project Structure

```
ai-insurance-claim-assistant/
├── data/
│   ├── raw/                        # Original anonymized claim samples
│   ├── synthetic/
│   │   ├── generate_claims.py      # Synthetic data generator (Faker)
│   │   ├── claims.json             # 100 generated claim records
│   │   └── claims.csv              # Same data in CSV format
│   ├── processed/
│   │   └── ragas_results.json      # RAGAS evaluation output
│   └── glossary/
│       ├── insurance_terms.py      # Glossary + FAQ knowledge base
│       └── insurance_terms.md      # Human-readable glossary reference
│
├── src/
│   ├── agent/
│   │   ├── claim_agent.py          # LangGraph agent (4-node graph)
│   │   └── ragas_eval.py           # RAGAS evaluation suite
│   ├── api/
│   │   ├── claim_api.py            # Claim data fetch layer (stub → extend)
│   │   └── query_handler.py        # Query routing (stub → extend)
│   ├── ui/
│   │   └── app.py                  # Streamlit chat UI
│   └── utils/
│       ├── data_generator.py       # Data generation utilities
│       ├── preprocessor.py         # Data cleaning & standardisation
│       └── anonymizer.py           # PII anonymisation helpers
│
├── config/
│   ├── settings.py                 # App configuration
│   └── model_config.json           # Model parameters
│
├── docs/
│   ├── architecture.md             # System design & diagrams
│   └── api_reference.md            # API endpoint documentation
│
├── tests/
│   ├── test_agent.py               # Agent unit tests
│   └── test_api.py                 # API integration tests
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA on synthetic claims
│   └── 02_synthetic_data_gen.ipynb # Data generation experiments
│
├── requirements.txt
├── run.sh
└── README.md
```

---

## 🤖 Agent Architecture (LangGraph)

The core agent is a **directed state graph** with 4 nodes:

```
User Input
    │
    ▼
┌─────────────────────┐
│  intent_classifier  │  ← Detects intent + extracts Claim ID (Claude)
└─────────────────────┘
    │
    ├── Claim ID found? ──► ┌──────────────┐
    │                       │ claim_lookup  │  ← Fetches claim from DB
    │                       └──────────────┘
    │                               │
    ▼                               ▼
┌─────────────────────────────────────┐
│           rag_retriever             │  ← Semantic search on glossary + FAQ
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        response_generator           │  ← Synthesises plain-English reply
└─────────────────────────────────────┘
    │
    ▼
  Output
```

### Supported Intents
| Intent | Description |
|---|---|
| `claim_status` | Status of a specific claim by ID |
| `claim_explanation` | Explains a claim concept or process |
| `document_info` | Required documents for a claim type |
| `payment_info` | Payout timelines and amounts |
| `appeal_info` | How to appeal a rejected claim |
| `general_faq` | General insurance knowledge |
| `greeting` | Welcome and onboarding |

---

## 📊 RAGAS Evaluation

The assistant is evaluated using **RAGAS** across four metrics:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer stay true to the retrieved context? |
| **Answer Relevancy** | Does the answer address the actual question asked? |
| **Context Precision** | Is the retrieved context targeted and on-point? |
| **Context Recall** | Does the retrieved context cover the full ground truth? |

Results are saved to `data/processed/ragas_results.json` and viewable in the **Streamlit evaluation tab**.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Clone & Install

```bash
git clone https://github.com/your-org/ai-insurance-claim-assistant.git
cd ai-insurance-claim-assistant
pip install -r requirements.txt
```

### 3. Set Your API Key

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. Generate Synthetic Data

```bash
python data/synthetic/generate_claims.py
```

This creates `claims.json` and `claims.csv` with 100 realistic, anonymized insurance claims.

### 5. Run the App

```bash
bash run.sh
```

Or directly:
```bash
streamlit run src/ui/app.py
```

Open your browser at **http://localhost:8501**

---

## 💬 How to Use the UI

### Chat Tab
1. **Enter a Claim ID** in the sidebar (e.g. `CLM-207473`) — or click a sample claim
2. **Ask questions in natural language**, for example:
   - *"What is the status of my claim?"*
   - *"Why was my claim rejected?"*
   - *"What documents do I still need to submit?"*
   - *"When will I receive my payment?"*
3. The **Claim Details panel** on the right shows live claim data
4. Expand **Agent Details** under any response to see intent classification and metadata

### RAGAS Evaluation Tab
1. Click **Run RAGAS Evaluation**
2. Optionally check **Use live agent** to evaluate with real LLM responses
3. View scores per metric with letter grades (A–D)
4. Inspect the full Q&A evaluation dataset

---

## 📦 Synthetic Data Schema

Each generated claim contains:

| Field | Type | Description |
|---|---|---|
| `claim_id` | string | Unique identifier (e.g. `CLM-207473`) |
| `policy_number` | string | Associated policy number |
| `claim_type` | string | Auto / Health / Home / Life / Travel |
| `status` | string | Current claim status (10 possible states) |
| `status_detail` | string | Plain-English status explanation |
| `submitted_date` | date | Date the claim was filed |
| `last_updated` | date | Most recent update |
| `amount_claimed` | float | Amount requested |
| `amount_approved` | float | Amount approved (0 if pending) |
| `claimant_name` | string | Anonymized claimant name |
| `assigned_adjuster` | string | Adjuster name and phone |
| `documents_required` | list | Documents still needed |
| `rejection_reason` | string | Reason if rejected (null otherwise) |
| `estimated_resolution_days` | int | Business days to resolution |

---

## 🗺️ Roadmap / Next Steps

- [ ] Connect `src/api/claim_api.py` to a live database (PostgreSQL / MongoDB)
- [ ] Add `.env` loading via `python-dotenv` in `config/settings.py`
- [ ] Complete unit tests in `tests/`
- [ ] Add a `Dockerfile` for containerised deployment
- [ ] Expand RAGAS test dataset to 50+ questions
- [ ] Add conversation memory persistence (Redis / SQLite)
- [ ] Multilingual support (Hindi, Tamil, Bengali for Indian market)
- [ ] WhatsApp / SMS interface integration

---

## 👥 Team

Built for **AI Friday Season 2** — a rapid prototyping challenge to demonstrate AI's potential in simplifying insurance customer experiences.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
