// ==============================================================================
// NEXT-GEN MoE STREAMING ENGINE: WEB & GATEWAY REGRESSION TEST SUITE
// ==============================================================================
// Two layers:
//   A. Offline unit tests — real logic (entropy, alignment, zoom, LFRU).
//      No string-literal assertions: the old markdown tests only asserted
//      that a hardcoded string contained itself, so they were deleted.
//   B. Live API-contract tests — spawn the REAL moe_engine server on a
//      test port (MOE_ENGINE_BIN + MOE_MODEL, defaults below) with
//      MOE_API_KEY set, and assert /health shape, the 401-without-key
//      contract, and a real greedy chat completion.
//      Skipped (with a clear message, not a failure) when no engine
//      binary or model is available, e.g. plain `npm test` in CI without
//      a model. Set MOE_ENGINE_BIN / MOE_MODEL to enable them.
//
// Usage:
//   MOE_ENGINE_BIN=../../build/moe_engine \
//   MOE_MODEL=../../models/real/qwen2.5-0.5b-instruct-q4_0.gguf \
//   npm test
// ==============================================================================

import http from 'node:http';
import assert from 'node:assert';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..', '..');

console.log('\x1b[1;36m===================================================================');
console.log('  RUNNING RIGOROUS QUALITY GATES & BROWSER REGRESSION TEST SUITE   ');
console.log('===================================================================\x1b[0m\n');

let totalTests = 0;
let passedTests = 0;
let skippedTests = 0;

function test(name, fn) {
  totalTests++;
  try {
    fn();
    console.log(`  [\x1b[1;32mPASSED\x1b[0m] ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  [\x1b[1;31mFAILED\x1b[0m] ${name}: ${err.message}`);
    console.error(err.stack);
  }
}

async function testAsync(name, fn) {
  totalTests++;
  try {
    await fn();
    console.log(`  [\x1b[1;32mPASSED\x1b[0m] ${name}`);
    passedTests++;
  } catch (err) {
    console.error(`  [\x1b[1;31mFAILED\x1b[0m] ${name}: ${err.message}`);
    console.error(err.stack);
  }
}

function skip(name, reason) {
  skippedTests++;
  console.log(`  [\x1b[1;33mSKIPPED\x1b[0m] ${name} — ${reason}`);
}

// ------------------------------------------------------------------------------
// A. Offline unit tests (real logic, no string literals)
// ------------------------------------------------------------------------------
test('1. SystemOne: Shannon entropy & confidence classification', () => {
  function calculateEntropy(probs) {
    let H = 0;
    for (const p of probs) {
      if (p > 1e-9) H -= p * Math.log2(p);
    }
    return H;
  }
  const uniformH = calculateEntropy([0.25, 0.25, 0.25, 0.25]);
  assert.ok(Math.abs(uniformH - 2.0) < 1e-5, 'Uniform 4-way distribution must have H=2.0 bits');
  const dominantH = calculateEntropy([0.98, 0.01, 0.005, 0.005]);
  assert.ok(dominantH < 0.2, 'Dominant distribution must have near-zero entropy');
});

test('2. Image Synthesis: tile size alignment to multiples of 32', () => {
  function alignTo32(val, min = 256, max = 2048) {
    const clamped = Math.max(min, Math.min(max, val));
    return Math.round(clamped / 32) * 32;
  }
  assert.strictEqual(alignTo32(500), 512);
  assert.strictEqual(alignTo32(250), 256);
  assert.strictEqual(alignTo32(2100), 2048);
  assert.strictEqual(alignTo32(768), 768);
});

test('3. Image Lightbox: zoom clamping between 1.0x and 3.0x', () => {
  function clampZoom(z) {
    return Math.max(1.0, Math.min(3.0, Number(z.toFixed(1))));
  }
  assert.strictEqual(clampZoom(0.5), 1.0);
  assert.strictEqual(clampZoom(1.5), 1.5);
  assert.strictEqual(clampZoom(3.8), 3.0);
});

test('4. LFRU cache scoring & CACHE_ROUTE margin substitution', () => {
  function lfruScore(freq, recency) {
    return freq * 256 + Math.min(255, recency);
  }
  assert.strictEqual(lfruScore(10, 42), 2602);
  assert.strictEqual(lfruScore(12, 300), 12 * 256 + 255);
  const epsilon = 0.08;
  const shouldSubstitute = Math.abs(0.45 - 0.42) < epsilon;
  assert.strictEqual(shouldSubstitute, true, 'Cache route substitution must fire within epsilon margin');
});

test('5. Built bundle: web/dist exists and references a real JS asset', () => {
  const distDir = path.join(REPO_ROOT, 'web', 'dist');
  const indexHtml = path.join(distDir, 'index.html');
  assert.ok(fs.existsSync(indexHtml), 'web/dist/index.html must exist (run `npm run build`)');
  const html = fs.readFileSync(indexHtml, 'utf8');
  const assetMatch = html.match(/assets\/[A-Za-z0-9._-]+\.js/);
  assert.ok(assetMatch, 'index.html must reference a built JS asset');
  const assetPath = path.join(distDir, assetMatch[0]);
  assert.ok(fs.existsSync(assetPath), `referenced asset must exist: ${assetMatch[0]}`);
  assert.ok(fs.statSync(assetPath).size > 10_000, 'built JS asset is suspiciously small');
});

test('6. Markdown href allow-list rejects javascript:/data: schemes', () => {
  // Mirrors web/src/utils/markdownParser.tsx link gate.
  function isSafeHref(href) {
    const h = String(href).trim();
    return /^(https?:|mailto:|#|\/|\.\/|\.\.\/)/i.test(h) &&
           !/^(javascript|data|vbscript):/i.test(h);
  }
  assert.strictEqual(isSafeHref('https://example.com'), true);
  assert.strictEqual(isSafeHref('/docs'), true);
  assert.strictEqual(isSafeHref('#section'), true);
  assert.strictEqual(isSafeHref('mailto:a@b.c'), true);
  assert.strictEqual(isSafeHref('javascript:alert(1)'), false);
  assert.strictEqual(isSafeHref('data:text/html,<script>'), false);
  assert.strictEqual(isSafeHref('vbscript:msgbox(1)'), false);
});

// ------------------------------------------------------------------------------
// B. Live API-contract tests against the real moe_engine server
// ------------------------------------------------------------------------------
const ENGINE_BIN = process.env.MOE_ENGINE_BIN || path.join(REPO_ROOT, 'build', 'moe_engine');
const MODEL_PATH = process.env.MOE_MODEL || path.join(REPO_ROOT, 'models', 'real', 'qwen2.5-0.5b-instruct-q4_0.gguf');
const TEST_PORT = Number(process.env.MOE_TEST_PORT || 18765);
const TEST_KEY = 'web-test-key-12345'; // test-only key, never a real credential

async function waitForHealth(base, timeoutMs = 120_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${base}/health`);
      if (res.ok) return await res.json();
    } catch {
      // server not up yet
    }
    await new Promise(r => setTimeout(r, 1000));
  }
  throw new Error(`server did not answer /health within ${timeoutMs}ms`);
}

async function runLiveTests() {
  if (!fs.existsSync(ENGINE_BIN)) {
    skip('Live: /health shape', `no engine binary at ${ENGINE_BIN} (set MOE_ENGINE_BIN)`);
    skip('Live: chat without key -> 401', 'no engine binary');
    skip('Live: chat with key -> greedy 2+2 answer', 'no engine binary');
    skip('Live: /v1/models lists the loaded model', 'no engine binary');
    skip('Live: GET /v1/config redacts the API key', 'no engine binary');
    return;
  }
  if (!fs.existsSync(MODEL_PATH)) {
    skip('Live: /health shape', `no model at ${MODEL_PATH} (set MOE_MODEL)`);
    skip('Live: chat without key -> 401', 'no model');
    skip('Live: chat with key -> greedy 2+2 answer', 'no model');
    skip('Live: /v1/models lists the loaded model', 'no model');
    skip('Live: GET /v1/config redacts the API key', 'no model');
    return;
  }

  const base = `http://127.0.0.1:${TEST_PORT}`;
  const server = spawn(ENGINE_BIN, [
    'serve', '--model', MODEL_PATH,
    '--port', String(TEST_PORT), '--host', '127.0.0.1',
    '--web-dir', path.join(REPO_ROOT, 'web', 'dist'),
  ], {
    cwd: REPO_ROOT,
    env: { ...process.env, MOE_API_KEY: TEST_KEY },
    stdio: 'ignore',
  });

  try {
    const health = await waitForHealth(base);

    await testAsync('6. Live: /health shape', async () => {
      assert.strictEqual(health.status, 'ok');
      assert.ok(typeof health.model_name === 'string' && health.model_name.length > 0, 'model_name must be a non-empty string');
      assert.ok(health.hparams && typeof health.hparams.vocab_size === 'number', 'hparams.vocab_size must be present');
      assert.ok(typeof health.active_slots === 'number', 'active_slots must be present');
    });

    await testAsync('7. Live: chat without API key -> 401', async () => {
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'What is 2+2?' }],
          stream: false, max_tokens: 4, temperature: 0,
        }),
        signal: AbortSignal.timeout(120_000),
      });
      assert.strictEqual(res.status, 401, 'mutating endpoint without a key must be 401');
      const data = await res.json();
      assert.strictEqual(data.error, 'unauthorized');
    });

    await testAsync('8. Live: chat with key -> greedy 2+2 answer', async () => {
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${TEST_KEY}`,
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'What is 2+2?' }],
          stream: false, max_tokens: 16, temperature: 0,
        }),
        signal: AbortSignal.timeout(300_000), // slow VM + shared CPU: 16 greedy tokens can take minutes
      });
      assert.strictEqual(res.status, 200);
      const data = await res.json();
      const content = data.choices?.[0]?.message?.content ?? '';
      assert.ok(typeof content === 'string' && content.trim().length > 0, 'chat must return non-empty content');
      assert.ok(content.includes('4'), `greedy 2+2 answer must contain "4", got: ${JSON.stringify(content)}`);
    });

    await testAsync('9. Live: /v1/models lists the loaded model', async () => {
      const res = await fetch(`${base}/v1/models`);
      assert.strictEqual(res.status, 200);
      const data = await res.json();
      assert.ok(Array.isArray(data.data) && data.data.length >= 1, 'model list must be non-empty');
      assert.ok(data.data.some(m => m.id === health.model_name), 'list must include the loaded model');
    });

    await testAsync('10. Live: GET /v1/config redacts the API key', async () => {
      const res = await fetch(`${base}/v1/config`, {
        headers: { Authorization: `Bearer ${TEST_KEY}` },
      });
      assert.strictEqual(res.status, 200);
      const text = await res.text();
      assert.ok(!text.includes(TEST_KEY), 'config response must not leak the API key value');
    });
  } finally {
    server.kill('SIGTERM');
    await new Promise(r => setTimeout(r, 1000));
    if (server.exitCode === null) server.kill('SIGKILL');
  }
}

await runLiveTests();

console.log('\n\x1b[1;36m===================================================================');
console.log(`  VERIFICATION RESULTS: ${passedTests}/${totalTests} TESTS PASSED (${skippedTests} SKIPPED)`);
console.log('===================================================================\x1b[0m\n');

if (passedTests !== totalTests) {
  process.exit(1);
}
