import requests
import json

# Ground truth labelled test cases
# label: what the correct verdict SHOULD be
TEST_CASES = [
    # ── VERIFIED cases (correct claims) ──────────────────────────────
    {
        "query": "what is the refund policy",
        "response": "The refund policy is 20 days.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the refund policy",
        "response": "Customers have 20 days to request a refund.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the federal minimum wage",
        "response": "The federal minimum wage is $7.25 per hour under FLSA.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the federal minimum wage",
        "response": "Minimum wage is $7.25 for 1hr of work.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the GDPR erasure deadline",
        "response": "Under GDPR Article 17 organizations must delete data within 30 days.",
        "label": "VERIFIED"
    },
    {
        "query": "what does HIPAA require for breach notification",
        "response": "HIPAA requires breach notification within 60 days of discovery.",
        "label": "VERIFIED"
    },
    {
        "query": "what are GDPR fines",
        "response": "GDPR fines can reach 20 million euros or 4 percent of annual turnover.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the ADA employee threshold",
        "response": "ADA applies to employers with 15 or more employees.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the server room temperature",
        "response": "The server room must be kept at 18 degrees Celsius.",
        "label": "VERIFIED"
    },
    {
        "query": "what is the CCPA response deadline",
        "response": "Businesses must respond to CCPA requests within 45 days.",
        "label": "VERIFIED"
    },

    # ── HALLUCINATION cases (wrong claims) ───────────────────────────
    {
        "query": "what is the refund policy",
        "response": "The refund policy is 25 days.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the refund policy",
        "response": "The refund policy is 30 days.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the federal minimum wage",
        "response": "The federal minimum wage is $9.00 per hour.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the federal minimum wage",
        "response": "Minimum wage is $15 per hour under FLSA.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the GDPR erasure deadline",
        "response": "GDPR requires data deletion within 7 days of a valid request.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what does HIPAA require for breach notification",
        "response": "HIPAA requires notification within 30 days.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what are GDPR fines",
        "response": "GDPR fines can reach 10 million euros maximum.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the ADA employee threshold",
        "response": "ADA applies to all employers regardless of size.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the server room temperature",
        "response": "The server room must be kept at 25 degrees Celsius.",
        "label": "HALLUCINATION_DETECTED"
    },
    {
        "query": "what is the CCPA response deadline",
        "response": "Businesses must respond to CCPA requests within 30 days.",
        "label": "HALLUCINATION_DETECTED"
    },

    # ── NEUTRAL cases (unverifiable) ──────────────────────────────────
    {
        "query": "what is the weather today",
        "response": "It is sunny and 25 degrees outside.",
        "label": "NEUTRAL"
    },
    {
        "query": "who founded the company",
        "response": "The company was founded in 2010.",
        "label": "NEUTRAL"
    },
]

def run():
    correct = 0
    wrong   = 0
    results = []

    for i, tc in enumerate(TEST_CASES):
        try:
            r = requests.post(
                "http://localhost:8000/verify",
                json={"query": tc["query"], "response": tc["response"]},
                timeout=30
            )
            got = r.json().get("status", "ERROR")
        except Exception as e:
            got = "ERROR"

        expected = tc["label"]
        passed   = got == expected
        correct += int(passed)
        wrong   += int(not passed)

        results.append({
            "passed":   passed,
            "expected": expected,
            "got":      got,
            "query":    tc["query"][:40],
            "response": tc["response"][:50],
        })

        mark = "✓" if passed else "✗"
        print(f"[{i+1:02d}] {mark}  expected={expected:<25} got={got:<25}  '{tc['response'][:45]}'")

    total    = len(TEST_CASES)
    accuracy = (correct / total) * 100

    print(f"\n{'─'*70}")
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct}")
    print(f"  Wrong:    {wrong}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"{'─'*70}")

    wrong_cases = [r for r in results if not r["passed"]]
    if wrong_cases:
        print(f"\nFailed cases:")
        for r in wrong_cases:
            print(f"  expected={r['expected']} got={r['got']}")
            print(f"    query:    {r['query']}")
            print(f"    response: {r['response']}")

if __name__ == "__main__":
    run()