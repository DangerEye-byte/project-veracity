from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import re

# ---------------------------------------------------------------------------
# Models — loaded once at startup
# ---------------------------------------------------------------------------
nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=-1,   # CPU; set to 0 for GPU
    top_k=None
)

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Thresholds — derived from exhaustive case analysis
# ---------------------------------------------------------------------------
QUERY_DOC_RELEVANCE    = 0.30   # query↔doc: below → wrong doc retrieved → NEUTRAL
ENTAILMENT_THRESHOLD   = 0.50   # NLI entailment: above → VERIFIED
CONTRADICTION_THRESHOLD= 0.90   # NLI contradiction: above → CONTRADICTION
AMBIGUITY_CEILING      = 0.35   # both NLI scores below → claim unverifiable → NEUTRAL

# Filler words stripped when checking if a claim is meaningful
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
    """Count real content words (≥3 chars, not filler) in text."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return len([w for w in words if w not in _FILLERS])

def _embed_similarity(text1: str, text2: str) -> float:
    emb = similarity_model.encode([text1, text2], convert_to_tensor=True)
    return util.cos_sim(emb[0], emb[1]).item()

def _nli_scores(premise: str, hypothesis: str) -> dict:
    results = nli(f"{premise} [SEP] {hypothesis}")[0]
    return {r["label"].upper(): r["score"] for r in results}

# ---------------------------------------------------------------------------
# Main auditor
# ---------------------------------------------------------------------------
def audit_response(truth: str, ai_claim: str, query: str = "") -> str:
    """
    Returns one of: VERIFIED | CONTRADICTION | NEUTRAL

    truth    : ground truth sentence from knowledge base
    ai_claim : the AI response claim to verify
    query    : the original user query (used for doc relevance check)
    """

    # ------------------------------------------------------------------
    # Guard: no truth available
    # ------------------------------------------------------------------
    if not truth or truth.strip() == "No reference found.":
        return "NEUTRAL"

    try:

        # ------------------------------------------------------------------
        # Step 1 — Is the retrieved document actually about the query topic?
        # Compares query↔truth, NOT claim↔truth.
        # The retriever already matched query→doc, but keyword overlap in
        # main.py can pass loosely related docs. This is the safety net.
        # ------------------------------------------------------------------
        if query and query.strip():
            query_doc_sim = _embed_similarity(query, truth)
            print(f"DEBUG - Query↔Doc  similarity : {query_doc_sim:.3f}")
            if query_doc_sim < QUERY_DOC_RELEVANCE:
                print("DEBUG - Retrieved doc is not relevant to query → NEUTRAL")
                return "NEUTRAL"

        # ------------------------------------------------------------------
        # Step 2 — Is the claim making any real assertion?
        # Catches: "xyz", "something", purely placeholder text.
        # A claim needs ≥2 meaningful content words to be verifiable.
        # ------------------------------------------------------------------
        claim_word_count = _meaningful_word_count(ai_claim)
        print(f"DEBUG - Claim meaningful word count: {claim_word_count}")
        if claim_word_count < 2:
            print("DEBUG - Claim has no meaningful content → NEUTRAL")
            return "NEUTRAL"

        # ------------------------------------------------------------------
        # Step 3 — NLI: does claim contradict or entail truth?
        # Forward only (truth as premise, claim as hypothesis).
        # Backward direction removed — unreliable, causes false VERIFIED.
        # ------------------------------------------------------------------
        scores = _nli_scores(truth, ai_claim)
        entailment_score    = scores.get("ENTAILMENT", 0)
        contradiction_score = scores.get("CONTRADICTION", 0)
        neutral_score       = scores.get("NEUTRAL", 0)

        print(f"DEBUG - NLI → entail={entailment_score:.3f}  "
              f"contra={contradiction_score:.3f}  "
              f"neutral={neutral_score:.3f}")

        # ------------------------------------------------------------------
        # Step 4 — Decision tree (order matters)
        # ------------------------------------------------------------------

        # 4a. Both signals weak → claim is too vague to produce a verdict
        #     e.g. "GDPR article 17 states that xyz", "federal wage exists"
        #     These are not wrong, just unverifiable → NEUTRAL
        if entailment_score < AMBIGUITY_CEILING and contradiction_score < AMBIGUITY_CEILING:
            print("DEBUG - Both NLI scores weak, claim is unverifiable → NEUTRAL")
            return "NEUTRAL"

        # 4b. Clear entailment → claim agrees with truth
        if entailment_score >= ENTAILMENT_THRESHOLD:
            print("DEBUG - Entailment dominant → VERIFIED")
            return "VERIFIED"

        # 4c. Very high contradiction → genuine factual error
        #     Threshold 0.90 chosen specifically to avoid flagging:
        #     - paraphrases (e.g. "$7.25 for 1hr" vs "$7.25 per hour")
        #     - partial info (claim is subset of truth)
        #     - informal phrasing
        #     Only fires when model is near-certain (0.90+)
        if contradiction_score >= CONTRADICTION_THRESHOLD:
            print("DEBUG - Contradiction dominant → CONTRADICTION")
            return "CONTRADICTION"

        # 4d. Ambiguous middle zone (contradiction 0.35–0.89, entailment 0.35–0.49)
        #     Claim is neither clearly wrong nor clearly right.
        #     Give benefit of the doubt — partial/incomplete info ≠ wrong.
        print("DEBUG - Ambiguous zone → VERIFIED (benefit of doubt)")
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
