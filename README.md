# Veracity

AI hallucinates. This fixes it.

VeracityVault sits between your AI and your users, intercepting responses, checking them against a knowledge base, and correcting wrong facts inline before anyone sees them. Not a warning banner. Not a disclaimer. An actual fix, with the source attached.

Built for a hackathon. Works on a laptop. No GPU needed.

---

## What it does

You ask something. Gemini answers. If it gets a fact wrong, wrong number of days, wrong dollar amount, wrong threshold then the wrong value gets struck through in red, the correct value appears in green, and hovering it shows you exactly which document it came from.

The whole thing happens in under 150ms and doesn't interrupt the streaming response.

---

## Stack

| Layer | What |
|---|---|
| Frontend | Vanilla HTML/CSS/JS - no framework |
| Node.js server | Express, port 3000, Gemini API proxy |
| AI model | Gemini 2.5 Flash Lite |
| Python backend | FastAPI + uvicorn, port 8000 |
| Retrieval | BM25 + Qdrant vector search via LangChain EnsembleRetriever |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Verification | `cross-encoder/nli-deberta-v3-small` (HuggingFace) |
| Vector DB | Qdrant in Docker |

---

## Project structure

```
project/
├── frontend/
│   ├── index.html          # landing page
│   └── chat.html           # demo UI
│
├── backend-node/
│   ├── server.js
│   ├── .env                # you modify this
│   └── routes/
│       ├── chat.js         # streams Gemini, calls auditor
│       └── verify.js       # proxies to FastAPI
│
└── app/
    ├── main.py             # FastAPI endpoints
    ├── database.py         # retriever setup
    └── auditor.py          # verification pipeline
```

---

## Setup

### 1. Start Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Python backend

```bash
pip install fastapi uvicorn langchain langchain-qdrant langchain-community \
    langchain-classic langchain-huggingface qdrant-client \
    sentence-transformers transformers torch rank-bm25 pydantic
```

```bash
# from the folder containing app/
uvicorn app.main:app --reload --port 8000
```

### 3. Node.js server

```bash
cd backend-node
# fill in GEMINI_API_KEY and FASTAPI_URL=http://localhost:8000
npm install
npm run dev
```

### 4. Open it

```
http://localhost:3000
```

---

## How the verification works

When Gemini finishes a response, it gets passed through a 5-stage pipeline:

1. **Hybrid retrieval** : BM25 and Qdrant vector search run in parallel, results merged via Reciprocal Rank Fusion. Catches both exact keyword matches and semantic matches.

2. **Relevance gates** : cosine similarity checks on query↔doc and claim↔doc. If either is too low, skip verification entirely. Keeps things fast.

3. **Numerical contradiction check** : NLI models treat "25 days" and "20 days" as semantically similar. We don't. A custom extractor pulls numbers with their unit type and flags mismatches directly.

4. **NLI classification** : DeBERTa scores entailment/contradiction/neutral. Contradiction threshold is 0.90 (not 0.50) : only flag when the model is confident.

5. **Paraphrase rule** : correct restatements score `neutral` in NLI, not `entailment`. We account for that: `neutral > 0.50 AND contradiction < 0.15 → VERIFIED`.

---

## Why not just ask GPT if it's true

We tried that first (llama3.2:3b via Ollama). It took 2-10 seconds per call, needed prompt engineering to output a single word, and still got it wrong sometimes. A 184MB discriminative classifier doing one job beats a generative LLM doing classification via prompting every time — 20-100x faster and no prompt babysitting.

---

## Known limitations

- Knowledge base is hardcoded in `main.py` and `chat.js`, swap the `KNOWLEDGE_BASE` list for your own documents
- NLI model struggles with very long responses , it sees the full response as one claim
- Works best on factual, numerical claims  vague statements like "GDPR is important" will return NEUTRAL
- Conversation history is browser memory only , clears on refresh

---

## Environment variables

```
GEMINI_API_KEY=your_key_here
FASTAPI_URL=http://localhost:8000
PORT=3000
NODE_ENV=development
```

---

## Running the accuracy benchmark

```bash
# make sure FastAPI is running
python test_accuracy.py
```

This runs 22 labelled test cases across verified, hallucination, and neutral categories and prints accuracy.
