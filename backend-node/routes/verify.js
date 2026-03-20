const express = require('express');
const router  = express.Router();

// ── POST /api/verify ───────────────────────────────────────────────────────
// Direct proxy to FastAPI /verify endpoint
// Body: { query: string, response: string }
// Used by the frontend when it wants to re-verify independently
// ──────────────────────────────────────────────────────────────────────────
router.post('/', async (req, res) => {
  const { query, response } = req.body;

  if (!query || !response) {
    return res.status(400).json({ error: 'query and response are required' });
  }

  try {
      const faRes = await fetch(`${process.env.FASTAPI_URL}/verify`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query, response }),
      signal:  AbortSignal.timeout(15000)
    });

    if (!faRes.ok) {
      const text = await faRes.text();
      console.error('[verify proxy error]', faRes.status, text);
      return res.status(502).json({ error: 'FastAPI returned an error', detail: text });
    }

    const data = await faRes.json();
    res.json(data);

  } catch (err) {
    console.error('[verify route error]', err.message);
    res.status(503).json({
      error:   'Auditor service unavailable',
      message: err.message
    });
  }
});

module.exports = router;