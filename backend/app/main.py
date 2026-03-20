from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from langchain_core.documents import Document
from app.database import initialize_retriever
from app.auditor import audit_response
import asyncio
import time
# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    Document(page_content="The company refund policy is strictly 20 days."),
    Document(page_content="The server room temperature must be 18°C."),
    Document(page_content="Under GDPR Article 17, individuals have the right to erasure ('right to be forgotten') and organizations must delete personal data within 30 days of a valid request."),
    Document(page_content="The statutory limitation period for filing a breach of contract claim is 6 years from the date the breach occurred under the Limitation Act 1980."),
    Document(page_content="Under the Americans with Disabilities Act (ADA), employers with 15 or more employees are required to provide reasonable accommodations to qualified individuals with disabilities."),
    Document(page_content="HIPAA requires covered entities to notify affected individuals of a data breach within 60 days of discovery. Breaches affecting more than 500 individuals must also be reported to the HHS Secretary."),
    Document(page_content="Under the Fair Labor Standards Act (FLSA), the federal minimum wage is $7.25 per hour. Overtime pay must be at least 1.5 times the regular rate for hours worked beyond 40 in a workweek."),
    Document(page_content="The Securities Exchange Act Section 10(b) and SEC Rule 10b-5 prohibit insider trading. Penalties include fines up to $5 million and imprisonment up to 20 years for individuals."),
    Document(page_content="Under the Foreign Corrupt Practices Act (FCPA), it is illegal for U.S. persons and companies to bribe foreign government officials to obtain or retain business. Civil penalties can reach $16,000 per violation."),
    Document(page_content="The General Data Protection Regulation (GDPR) imposes maximum fines of €20 million or 4% of annual global turnover, whichever is higher, for the most serious infringements."),
    Document(page_content="Under the Sarbanes-Oxley Act Section 302, CEOs and CFOs must personally certify the accuracy of financial statements. False certification carries criminal penalties of up to $5 million and 20 years imprisonment."),
    Document(page_content="The California Consumer Privacy Act (CCPA) grants consumers the right to know what personal data is collected, the right to delete it, and the right to opt out of its sale. Businesses have 45 days to respond to consumer requests."),
]

retriever = None

# ---------------------------------------------------------------------------
# Lifespan — initialize on startup, not at import time
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = initialize_retriever(KNOWLEDGE_BASE)
    yield

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class AuditRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    query: str
    response: str

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
_STOP_WORDS = {
    "the","a","an","is","in","of","to","was","what","when",
    "how","which","are","does","do","tell","me","about","give"
}

@app.post("/verify")
async def verify_flow(data: AuditRequest):
    t0 = time.perf_counter()
    relevant_docs = await asyncio.to_thread(retriever.invoke, data.query)

    if not relevant_docs:
        return {"status": "NEUTRAL", "message": "No relevant documents found."}

    context = relevant_docs[0].page_content

    query_words   = set(data.query.lower().split()) - _STOP_WORDS
    context_words = set(context.lower().split())
    overlap       = query_words & context_words
    print(f"DEBUG - Keyword overlap: {overlap}")

    if not overlap:
        return {"status": "NEUTRAL", "message": "Query does not match any knowledge base documents."}

    verdict = audit_response(context, data.response, query=data.query)
    print(f"DEBUG - Verdict: {verdict}")
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"[LATENCY] {latency_ms:.1f}ms")
    if verdict == "CONTRADICTION":
        return {
            "status": "HALLUCINATION_DETECTED",
            "correction": f"Actually, {context}",
            "reference": context
        }

    return {
        "status": verdict,
        "message": "Claim matches ground truth." if verdict == "VERIFIED"
                   else "Could not verify claim against knowledge base."
    }