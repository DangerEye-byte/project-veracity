
import requests

def audit_response(truth, ai_claim):
    if truth == "No reference found.":
        return "NEUTRAL"
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                # "model" : "phi4",
                "model": "llama3.2:3b",
                "stream": False,
                "options": {"temperature": 0.0},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict fact-checker. "
                            "Reply with ONE word only: CONTRADICT or ENTAIL. "
                            "CONTRADICTION if the claim disagrees with or is inconsistent with the truth. "
                            "ENTAILMENT if they agree. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"TRUTH: \"{truth}\"\n"
                            f"CLAIM: \"{ai_claim}\"\n\n"
                            "If CLAIM contradicts TRUTH → CONTRADICT\n"
                            "If they agree → ENTAIL\n\n"
                            "One word:"
                        )
                    }
                ]
            },
            timeout=30
        )
        raw = response.json()
        if "error" in raw:
            print(f"Ollama Error: {raw['error']}")
            return "NEUTRAL"
        result = raw.get("message", {}).get("content", "").strip().upper()
        print(f"DEBUG - LLM raw result: {result}")
        return "CONTRADICTION" if "CONTRADICT" in result else "VERIFIED"
    except Exception as e:
        print(f"Auditor Error: {e}")
        return "NEUTRAL"