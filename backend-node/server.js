require('dotenv').config();
const express    = require('express');
const cors       = require('cors');
const path       = require('path');
const chatRoute  = require('./routes/chat');
const verifyRoute = require('./routes/verify');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Validate required env vars on startup ──────────────────────────────────
const REQUIRED_ENV = ['GEMINI_API_KEY', 'FASTAPI_URL'];
const missing = REQUIRED_ENV.filter(k => !process.env[k]);
if (missing.length) {
  console.error(`[FATAL] Missing env vars: ${missing.join(', ')}`);
  console.error('[FATAL] Copy .env.example to .env and fill in your values.');
  process.exit(1);
}

// ── Middleware ─────────────────────────────────────────────────────────────
app.use(cors({
  origin: [`http://localhost:${PORT}`, 'http://127.0.0.1:' + PORT],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
app.use(express.json({ limit: '1mb' }));

// ── Static frontend ────────────────────────────────────────────────────────
// Serves index.html, chat.html, and any assets from /frontend
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// ── API Routes ─────────────────────────────────────────────────────────────
app.use('/api/chat',   chatRoute);
app.use('/api/verify', verifyRoute);

// ── Health check ───────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    node: process.version,
    fastapi: process.env.FASTAPI_URL,
    timestamp: new Date().toISOString()
  });
});

// ── 404 fallback for unknown API routes ───────────────────────────────────
app.use('/api/*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// ── Global error handler ──────────────────────────────────────────────────
app.use((err, req, res, next) => {
  console.error('[ERROR]', err.message);
  res.status(500).json({ error: 'Internal server error' });
});

// ── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n┌─────────────────────────────────────────┐`);
  console.log(`│  VeracityVault Node Server               │`);
  console.log(`│  http://localhost:${PORT}                   │`);
  console.log(`│  FastAPI target: ${process.env.FASTAPI_URL}  │`);
  console.log(`└─────────────────────────────────────────┘\n`);
});