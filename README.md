# Insurance Chatbot (Korean) — Query Classifier + Agentic RAG

A Korean-language insurance customer-support chatbot built around two components:
a 26-label query classifier and a LangChain / LangGraph RAG pipeline grounded in
the *한화 100세 암치료보장보험* (Hanwha 100-year Cancer Treatment Insurance)
product document.

The chatbot classifies each user query into a fixed taxonomy of intents (greeting,
PII, coverage, claims, etc.), then either replies from a static template or runs
retrieval-augmented generation over the policy document. A LangGraph extension
adds a CRAG-style corrective loop on top of the base pipeline.

## Files

| File | Description |
| ---- | ----------- |
| `Insurance_query_classifier.ipynb` | Query classifier notebook with two backends: GPT-4o-mini (OpenAI structured outputs) and Llama-3-Korean-Bllossom-8B (constrained decoding via `outlines` + few-shot Chain-of-Thought prompting). |
| `classifier.py` | Module version of the GPT-4o-mini classifier, imported by the chatbot notebook. |
| `Insurance_chatbot_langchain.ipynb` | End-to-end chatbot. Builds a LangChain RAG pipeline first (PDF parsing → chunking → label-aware indexing → Chroma vector store), then extends it into an agentic LangGraph pipeline with relevance grading, query rewriting, and hybrid retrieval. |

## Pipeline overview

Each query flows through the following stages:

1. **Classification** — heuristic regex pre-filter (PII patterns, greetings,
   human-handoff phrases) first; otherwise an LLM call with a Pydantic schema
   forces structured output into one of 26 labels.
2. **Routing** — static-template reply for conversational / safety labels
   (`GREET`, `GOODBYE`, `PRIVACY_PII`, `MEDICAL_DIAGNOSIS_ADVICE`, ...) or RAG
   for insurance-content labels (`COVERAGE_BENEFITS`, `CLAIM_FILING`,
   `PREMIUM_STABILITY_RENEWAL`, ...).
3. **Retrieval** — dense search on Chroma (bge-m3 embeddings) with a metadata
   filter aligned to the classified label, so retrieval is both
   vector-similarity- and category-constrained.
4. **Grading (agentic extension)** — a cross-encoder reranker
   (`bge-reranker-v2-m3`) scores each chunk against the query. Chunks below a
   per-chunk threshold are dropped; if the average score is too low, the graph
   loops back to rewrite.
5. **Query rewriting (agentic extension)** — retry 1 applies HyDE + jargon /
   synonym expansion with the metadata filter preserved; retry 2 drops the
   filter and switches to a BM25 + dense hybrid search merged via Reciprocal
   Rank Fusion. Bounded at 2 retries.
6. **Generation** — GPT-4o-mini answers from the graded chunks only, with a
   label-specific formatting hint.

## Label taxonomy (26 total)

- **Conversation control (5):** `GREET`, `GOODBYE`, `FEEDBACK_POS`,
  `FEEDBACK_NEG`, `HUMAN_HANDOFF`
- **Pre-purchase (13):** `PRODUCT_OVERVIEW`, `PLAN_OPTIONS`,
  `COVERAGE_BENEFITS`, `EXCLUSIONS_LIMITATIONS`, `WAITING_PERIOD`,
  `PREEXISTING_CONDITIONS`, `ELIGIBILITY_UNDERWRITING`,
  `PREMIUM_STABILITY_RENEWAL`, `PRICING_GENERAL`, `QUOTE_PERSONALIZED`,
  `PAYMENT_BILLING`, `COORDINATION_OTHER_INSURANCE`, `APPLICATION_HOWTO`
- **Claims / servicing (5):** `CLAIM_FILING`, `CLAIM_DOCUMENTS`,
  `CLAIM_STATUS_TIMELINE`, `POLICY_CHANGES_CANCEL_RENEW`, `COMPLAINT_DISPUTE`
- **Safety routing (3):** `PRIVACY_PII`, `MEDICAL_DIAGNOSIS_ADVICE`,
  `OUT_OF_SCOPE`

Conversation and safety labels are routed to static replies. The remaining
labels are routed through RAG.

## Setup

The notebooks are written for **Google Colab** (GPU recommended for the Llama
backend and the cross-encoder reranker). An OpenAI API key is required for the
GPT-4o-mini paths.

Main dependencies:

- `openai`, `pydantic`
- `langchain`, `langchain-community`, `langchain-openai`, `langchain-huggingface`
- `langgraph`
- `chromadb`, `sentence-transformers`, `rank-bm25`
- `pymupdf`, `langchain-pymupdf4llm`
- `outlines`, `transformers`, `accelerate` *(Llama path only)*

Set `OPENAI_API_KEY` in the Colab `userdata` store (or as an environment
variable) before running the notebooks. The Llama path additionally requires a
Hugging Face login.

## Notes

- The reference document used for RAG is the publicly available
  [한화 100세 암치료보장보험 상품요약서](https://www.hwgeneralins.com/notice/ir/product-ing01.do).
- Chunks are labeled by the same 26-label taxonomy used for query
  classification, which lets retrieval be filtered to the matching category
  rather than relying on cosine similarity alone.
- The LangGraph CRAG extension was added because the one-shot LangChain
  pipeline produced incorrect answers on a few queries (e.g.
  *"상품 종류(종)는 어떻게 나뉘나요?"*); the corrective loop fixes these.
- The Llama classifier uses `outlines` constrained decoding, which guarantees
  the output is a schema-valid `QueryClassification` JSON object and removes
  the parse-failure / invalid-label failure modes of free-form generation.
