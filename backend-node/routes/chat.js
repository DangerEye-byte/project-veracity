const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const router  = express.Router();

// Initialise Gemini client once — key never leaves the server
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// System prompt designed to produce confident answers that may hallucinate
// specific numbers/dates — perfect for demonstrating VeracityVault
const SYSTEM_PROMPT = `You are a helpful legal and compliance assistant.
Answer questions about regulations, company policies, and legal requirements confidently.
Give specific numbers, dates, and figures in your answers but make them specifically wrong slighly like change the figures etc in every answer.
Keep answers to 2-3 sentences. Be direct and authoritative.
Do not hedge or say "I'm not sure" — always give a specific answer.
use the following data for reference and answer only from this:

"The company refund policy is strictly 20 days.",
"The server room temperature must be 18°C.",
"Under GDPR Article 17, individuals have the right to erasure ('right to be forgotten') and organizations must delete personal data within 30 days of a valid request.",
"The statutory limitation period for filing a breach of contract claim is 6 years from the date the breach occurred under the Limitation Act 1980.",
"Under the Americans with Disabilities Act (ADA), employers with 15 or more employees are required to provide reasonable accommodations to qualified individuals with disabilities.",
"HIPAA requires covered entities to notify affected individuals of a data breach within 60 days of discovery. Breaches affecting more than 500 individuals must also be reported to the HHS Secretary.",
"Under the Fair Labor Standards Act (FLSA), the federal minimum wage is $7.25 per hour. Overtime pay must be at least 1.5 times the regular rate for hours worked beyond 40 in a workweek.",
"The Securities Exchange Act Section 10(b) and SEC Rule 10b-5 prohibit insider trading. Penalties include fines up to $5 million and imprisonment up to 20 years for individuals.",
"Under the Foreign Corrupt Practices Act (FCPA), it is illegal for U.S. persons and companies to bribe foreign government officials to obtain or retain business. Civil penalties can reach $16,000 per violation.",
"The General Data Protection Regulation (GDPR) imposes maximum fines of €20 million or 4% of annual global turnover, whichever is higher, for the most serious infringements.",
"Under the Sarbanes-Oxley Act Section 302, CEOs and CFOs must personally certify the accuracy of financial statements. False certification carries criminal penalties of up to $5 million and 20 years imprisonment.",
"The California Consumer Privacy Act (CCPA) grants consumers the right to know what personal data is collected, the right to delete it, and the right to opt out of its sale. Businesses have 45 days to respond to consumer requests.",

`;

// ── POST /api/chat ─────────────────────────────────────────────────────────
// Body: { query: string, history: Array<{role: string, text: string}> }
//
// history is the full prior conversation EXCLUDING the current query.
// Each entry: { role: 'user' | 'model', text: string }
//
// Gemini contents format:
//   [ { role: 'user', parts: [{text}] }, { role: 'model', parts: [{text}] }, ... ]
// ──────────────────────────────────────────────────────────────────────────
router.post('/', async (req, res) => {
  const { query, history = [] } = req.body;

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'query is required' });
  }

  // Validate history — must be array of {role, text}
  const safeHistory = Array.isArray(history)
    ? history.filter(m => m && typeof m.text === 'string' && ['user','model'].includes(m.role))
    : [];

  // ── SSE headers ────────────────────────────────────────────────────────
  res.setHeader('Content-Type',      'text/event-stream');
  res.setHeader('Cache-Control',     'no-cache');
  res.setHeader('Connection',        'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  const send = (event, data) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const model = genAI.getGenerativeModel({
      model: 'gemini-2.5-flash-lite',
      systemInstruction: SYSTEM_PROMPT,
      generationConfig: {
        temperature:     0.9,
        maxOutputTokens: 200,
        topP:            0.95,
      }
    });

    // ── Build Gemini contents array ────────────────────────────────────
    // Format: alternating user/model turns + current query at the end.
    // Gemini requires the array to start with 'user' and alternate strictly.
    const contents = [
      // Prior turns from history
      ...safeHistory.map(m => ({
        role:  m.role,
        parts: [{ text: m.text }]
      })),
      // Current user message
      {
        role:  'user',
        parts: [{ text: query }]
      }
    ];

    console.log(`[CHAT] Sending ${contents.length} turns to Gemini (${safeHistory.length} history + 1 new)`);

    const streamResult = await model.generateContentStream({ contents });

    let fullText = '';

    for await (const chunk of streamResult.stream) {
      const token = chunk.text();
      if (token) {
        fullText += token;
        send('token', { text: token });
      }
    }

    send('done', { fullText });

    // ── Call FastAPI auditor ───────────────────────────────────────────
    console.log(`[CHAT] Calling FastAPI at ${process.env.FASTAPI_URL}/verify`);
    const verdict = await callFastAPIVerify(query, fullText);
    console.log(`[CHAT] FastAPI verdict:`, JSON.stringify(verdict));
    send('verdict', verdict);

  } catch (err) {
    console.error('[chat route error]', err.message);
    send('error', { message: err.message || 'Gemini request failed' });
  } finally {
    res.end();
  }
});

// ── Internal: call FastAPI /verify ────────────────────────────────────────
async function callFastAPIVerify(query, response) {
  try {
    const res = await fetch(`${process.env.FASTAPI_URL}/verify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, response }),
      signal:  AbortSignal.timeout(15000)
    });

    if (!res.ok) throw new Error(`FastAPI returned ${res.status}`);
    return await res.json();

  } catch (err) {
    console.error('[FastAPI verify error]', err.message);
    return {
      status:  'NEUTRAL',
      message: `Auditor unavailable: ${err.message}`
    };
  }
}

module.exports = router;