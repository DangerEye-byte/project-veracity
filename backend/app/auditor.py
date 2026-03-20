# from transformers import pipeline
# from sentence_transformers import util
# from app.database import shared_embedding_model   # ← shared, not reloaded
# import re

# # ---------------------------------------------------------------------------
# # NLI model — only new model loaded here
# # ---------------------------------------------------------------------------
# nli = pipeline(
#     "text-classification",
#     model="cross-encoder/nli-deberta-v3-small",
#     device=-1,
#     top_k=None
# )

# # ---------------------------------------------------------------------------
# # Thresholds
# # ---------------------------------------------------------------------------
# QUERY_DOC_RELEVANCE     = 0.30
# ENTAILMENT_THRESHOLD    = 0.50
# CONTRADICTION_THRESHOLD = 0.90
# AMBIGUITY_CEILING       = 0.35

# _FILLERS = {
#     'the','and','for','that','this','with','from','are','was','were',
#     'has','have','been','its','their','about','states','says','claims',
#     'according','under','which','what','how','when','where','who','is',
#     'not','can','will','must','should','may','also','only','any','all'
# }

# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def _meaningful_word_count(text: str) -> int:
#     words = re.findall(r"[a-zA-Z]{3,}", text.lower())
#     return len([w for w in words if w not in _FILLERS])

# def _embed_similarity(text1: str, text2: str) -> float:
#     emb = shared_embedding_model.encode(   # ← same instance as database.py
#         [text1, text2], convert_to_tensor=True
#     )
#     return util.cos_sim(emb[0], emb[1]).item()

# def _nli_scores(premise: str, hypothesis: str) -> dict:
#     results = nli(f"{premise} [SEP] {hypothesis}")[0]
#     return {r["label"].upper(): r["score"] for r in results}

# # ---------------------------------------------------------------------------
# # Main auditor
# # ---------------------------------------------------------------------------
# def audit_response(truth: str, ai_claim: str, query: str = "") -> str:
#     if not truth or truth.strip() == "No reference found.":
#         return "NEUTRAL"

#     try:
#         # Step 1 — Is retrieved doc actually about the query?
#         if query and query.strip():
#             query_doc_sim = _embed_similarity(query, truth)
#             print(f"DEBUG - Query↔Doc  similarity : {query_doc_sim:.3f}")
#             if query_doc_sim < QUERY_DOC_RELEVANCE:
#                 print("DEBUG - Retrieved doc not relevant to query → NEUTRAL")
#                 return "NEUTRAL"

#         # Step 2 — Is the claim making a real assertion?
#         if _meaningful_word_count(ai_claim) < 2:
#             print("DEBUG - Claim has no meaningful content → NEUTRAL")
#             return "NEUTRAL"

#         # Step 3 — NLI
#         scores              = _nli_scores(truth, ai_claim)
#         entailment_score    = scores.get("ENTAILMENT", 0)
#         contradiction_score = scores.get("CONTRADICTION", 0)
#         neutral_score       = scores.get("NEUTRAL", 0)

#         print(f"DEBUG - NLI → entail={entailment_score:.3f}  "
#               f"contra={contradiction_score:.3f}  "
#               f"neutral={neutral_score:.3f}")

#         # Step 4 — Decision
#         if entailment_score < AMBIGUITY_CEILING and contradiction_score < AMBIGUITY_CEILING:
#             print("DEBUG - Claim unverifiable → NEUTRAL")
#             return "NEUTRAL"

#         if entailment_score >= ENTAILMENT_THRESHOLD:
#             return "VERIFIED"

#         if contradiction_score >= CONTRADICTION_THRESHOLD:
#             return "CONTRADICTION"

#         # Ambiguous — partial/informal info, benefit of doubt
#         return "VERIFIED"

#     except Exception as e:
#         print(f"Auditor Error: {e}")
#         return "NEUTRAL"





from transformers import pipeline
from sentence_transformers import util
from app.database import shared_embedding_model
import re

# ---------------------------------------------------------------------------
# NLI model
# ---------------------------------------------------------------------------
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,
    top_k=None
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
QUERY_DOC_RELEVANCE     = 0.30
CLAIM_DOC_RELEVANCE     = 0.25
CONTRADICTION_THRESHOLD = 0.90
ENTAILMENT_THRESHOLD    = 0.50

_FILLERS = {
    'the','and','for','that','this','with','from','are','was','were',
    'has','have','been','its','their','about','states','says','claims',
    'according','under','which','what','how','when','where','who','is',
    'not','can','will','must','should','may','also','only','any','all'
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _meaningful_word_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return len([w for w in words if w not in _FILLERS])

def _embed_similarity(text1: str, text2: str) -> float:
    emb = shared_embedding_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()

def _nli_scores(premise: str, hypothesis: str) -> dict:
    results = nli(f"{premise} [SEP] {hypothesis}")[0]
    return {r["label"].upper(): r["score"] for r in results}

def _extract_numbers(text: str) -> list:
    """
    Extract (value_float, unit_type) pairs from text.
    Unit types group numbers so we only compare apples to apples.
    """
    results = []
    seen = set()

    patterns = [
        (r'[\$€£¥]?\s*(\d+(?:\.\d+)?)\s*(million|billion)', 'currency_large'),
        (r'\$\s*(\d+(?:\.\d+)?)',                            'dollar'),
        (r'€\s*(\d+(?:\.\d+)?)',                             'euro'),
        (r'(\d+(?:\.\d+)?)\s*%',                             'percent'),
        (r'(\d+(?:\.\d+)?)\s*°?\s*[CcFf]\b',                'temperature'),
        (r'(\d+(?:\.\d+)?)\s*(days?|hours?|years?|months?|weeks?)', 'time'),
        (r'(\d+(?:\.\d+)?)\s*x\b',                          'multiplier'),
        (r'\b(\d+(?:\.\d+)?)\b',                             'number'),
    ]

    for pattern, unit in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            pos = m.start()
            if pos in seen:
                continue
            seen.add(pos)
            try:
                val = float(m.group(1).replace(',', ''))
            except (ValueError, IndexError):
                continue
            if unit == 'currency_large':
                suffix = m.group(2).lower()
                val *= (1_000_000 if 'million' in suffix else 1_000_000_000)
                unit = 'currency'
            results.append((val, unit))

    return results


def _numerical_contradiction(truth: str, claim: str) -> bool:
    """
    Returns True if the claim contains a number that contradicts truth
    in the same unit category.

    NLI is blind to numbers — "25 days" vs "20 days" scores neutral=0.98.
    This function catches those mismatches directly.

    Fires on:   "25 days" vs "20 days"    → True  (contradiction)
                "$9/hr"   vs "$7.25/hr"   → True
                "25°C"    vs "18°C"        → True
    Safe on:    "$7.25"   vs "$7.25"       → False (same value)
                "30 days" + claim has no number → False
    """
    truth_nums = _extract_numbers(truth)
    claim_nums = _extract_numbers(claim)

    if not truth_nums or not claim_nums:
        return False

    # Group by unit type
    truth_by_unit = {}
    for val, unit in truth_nums:
        truth_by_unit.setdefault(unit, []).append(val)

    for c_val, c_unit in claim_nums:
        if c_unit not in truth_by_unit:
            continue
        truth_vals = truth_by_unit[c_unit]
        # If claim value doesn't match ANY truth value of same unit → contradiction
        if not any(abs(c_val - t_val) < 0.001 for t_val in truth_vals):
            print(f"DEBUG - Numerical mismatch: claim={c_val} ({c_unit}), "
                  f"truth has {truth_vals}")
            return True

    return False


# ---------------------------------------------------------------------------
# Main auditor
# ---------------------------------------------------------------------------
def audit_response(truth: str, ai_claim: str, query: str = "") -> str:
    """
    Returns: VERIFIED | CONTRADICTION | NEUTRAL

    NEUTRAL     → wrong doc, nonsense claim, or completely unrelated topic
    CONTRADICTION → claim has a factual error (numerical or logical)
    VERIFIED    → claim is correct, paraphrase, partial, or not provably wrong
    """

    if not truth or truth.strip() == "No reference found.":
        return "NEUTRAL"

    try:
        # Gate 1 — Nonsense/placeholder claim
        if _meaningful_word_count(ai_claim) < 2:
            print("DEBUG - Claim has no meaningful content → NEUTRAL")
            return "NEUTRAL"

        # Gate 2 — query↔doc: is the retrieved doc about what the user asked?
        if query and query.strip():
            q_sim = _embed_similarity(query, truth)
            print(f"DEBUG - Query↔Doc  similarity : {q_sim:.3f}")
            if q_sim < QUERY_DOC_RELEVANCE:
                print("DEBUG - Retrieved doc not relevant to query → NEUTRAL")
                return "NEUTRAL"

        # Gate 3 — claim↔doc: is the claim even about the same topic as the doc?
        # This catches cases where query was vague but claim is clearly off-topic.
        # Example: query="refund", wrong doc=FLSA, claim="refund is 25 days"
        #   → claim talks about refund, FLSA doc doesn't → low sim → NEUTRAL
        c_sim = _embed_similarity(ai_claim, truth)
        print(f"DEBUG - Claim↔Doc  similarity : {c_sim:.3f}")
        if c_sim < CLAIM_DOC_RELEVANCE:
            print("DEBUG - Claim not relevant to retrieved doc → NEUTRAL")
            return "NEUTRAL"

        # Numerical check — BEFORE NLI because NLI cannot detect number mismatches
        # "25 days" vs "20 days" → NLI says neutral=0.98, we say CONTRADICTION
        if _numerical_contradiction(truth, ai_claim):
            print("DEBUG - Numerical contradiction → CONTRADICTION")
            return "CONTRADICTION"

        # NLI check
        scores              = _nli_scores(truth, ai_claim)
        entailment_score    = scores.get("ENTAILMENT",    0)
        contradiction_score = scores.get("CONTRADICTION", 0)
        neutral_score       = scores.get("NEUTRAL",       0)

        print(f"DEBUG - NLI → entail={entailment_score:.3f}  "
              f"contra={contradiction_score:.3f}  "
              f"neutral={neutral_score:.3f}")

        # NLI is very confident it contradicts → CONTRADICTION
        if contradiction_score >= CONTRADICTION_THRESHOLD:
            print("DEBUG - High NLI contradiction → CONTRADICTION")
            return "CONTRADICTION"

        # NLI is confident it agrees → VERIFIED
        if entailment_score >= ENTAILMENT_THRESHOLD:
            print("DEBUG - High entailment → VERIFIED")
            return "VERIFIED"

        # Neutral-dominant + near-zero contradiction = correct paraphrase
        # NLI returns neutral (not entailment) for summaries and restatements.
        # contra≈0 confirms it's not wrong — just worded differently.
        # Example: truth="GDPR fines €20M or 4%"
        #          claim="Under GDPR, max fines are €20M or 4% of turnover"
        #          → entail=0.22, contra=0.001, neutral=0.77 → VERIFIED
        if neutral_score > 0.50 and contradiction_score < 0.15:
            print("DEBUG - Neutral-dominant + low contradiction → VERIFIED (paraphrase)")
            return "VERIFIED"

        # Benefit of the doubt — partial info, vague but not wrong
        print("DEBUG - Ambiguous → VERIFIED (benefit of doubt)")
        return "VERIFIED"

    except Exception as e:
        print(f"Auditor Error: {e}")
        return "NEUTRAL"




# =========================================================================== v2 - too strict ( not sure to implement)
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer, util
# import re

# nli = pipeline(
#     "text-classification",
#     model="cross-encoder/nli-deberta-v3-small",
#     device=-1,
#     top_k=None
# )

# similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Thresholds
# QUERY_DOC_RELEVANCE   = 0.30   # query vs retrieved doc — is doc actually about this topic?
# ENTAILMENT_THRESHOLD  = 0.50   # claim clearly agrees with truth
# CONTRADICTION_THRESHOLD = 0.90 # claim clearly disagrees with truth
# AMBIGUITY_CEILING     = 0.35   # if BOTH scores below this → claim is unverifiable/nonsense

# def is_meaningful_claim(claim: str) -> bool:
#     """
#     Reject claims that are too vague or contain no real assertions.
#     Catches: 'xyz', 'something', single words, placeholder text.
#     """
#     words = re.findall(r'[a-zA-Z]{3,}', claim.lower())
#     # Remove filler words
#     fillers = {'the','and','for','that','this','with','from','are','was',
#                'were','has','have','been','its','their','about','states',
#                'says','claims','according'}
#     meaningful = [w for w in words if w not in fillers]
#     print(f"DEBUG - Meaningful words in claim: {meaningful}")
#     return len(meaningful) >= 2  # needs at least 2 real words to be verifiable

# def audit_response(truth: str, ai_claim: str, query: str = "") -> str:
#     if not truth or truth.strip() == "No reference found.":
#         return "NEUTRAL"

#     try:
#         # Step 1 — Is the claim even making a real assertion?
#         if not is_meaningful_claim(ai_claim):
#             print("DEBUG - Claim is nonsense/unverifiable")
#             return "NEUTRAL"

#         # Step 2 — Is the retrieved document actually about the query topic?
#         # Use query vs truth (not claim vs truth) — retriever already matched these
#         # but keyword overlap in main.py can sometimes pass loosely related docs
#         if query:
#             embeddings = similarity_model.encode([query, truth], convert_to_tensor=True)
#             query_doc_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
#             print(f"DEBUG - Query↔Doc similarity: {query_doc_sim:.3f}")
#             if query_doc_sim < QUERY_DOC_RELEVANCE:
#                 print("DEBUG - Retrieved doc not relevant to query")
#                 return "NEUTRAL"

#         # Step 3 — NLI: does the claim contradict the truth?
#         results = nli(f"{truth} [SEP] {ai_claim}")[0]
#         scores = {r["label"].upper(): r["score"] for r in results}
#         print(f"DEBUG - NLI scores: {scores}")

#         entailment_score    = scores.get("ENTAILMENT", 0)
#         contradiction_score = scores.get("CONTRADICTION", 0)
#         neutral_score       = scores.get("NEUTRAL", 0)

#         # Both signals weak → claim is too vague to verify (e.g. "GDPR exists")
#         if entailment_score < AMBIGUITY_CEILING and contradiction_score < AMBIGUITY_CEILING:
#             print("DEBUG - Claim too vague/ambiguous to verify")
#             return "NEUTRAL"

#         # Clear entailment → verified
#         if entailment_score >= ENTAILMENT_THRESHOLD:
#             return "VERIFIED"

#         # Very high contradiction → genuine factual error
#         if contradiction_score >= CONTRADICTION_THRESHOLD:
#             return "CONTRADICTION"

#         # Moderate contradiction but not definitive → benefit of doubt
#         return "VERIFIED"

#     except Exception as e:
#         print(f"Auditor Error: {e}")
#         return "NEUTRAL"
