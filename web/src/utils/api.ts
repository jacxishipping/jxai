// Shared fetch wrapper for the engine API.
//
// When the server was started with MOE_API_KEY set, every mutating endpoint
// requires `Authorization: Bearer <key>` (or `x-api-key:`). The key is
// entered in Settings and kept ONLY in this module's memory for the current
// page load: it is never written to localStorage, sessionStorage, a cookie,
// or the bundle, and it is cleared on reload / tab close.
//
// Tradeoff, stated plainly: memory-only storage protects the key from
// persistence-based theft (another tab, a later XSS in a different page,
// disk inspection) and from surviving in browser storage after logout, but
// it does NOT protect against script code running in this same page while
// the key is loaded — any XSS in the app could still read it from memory,
// exactly as it could read it from localStorage. The mitigation for that
// is the same-origin-only server (no third-party scripts) plus keeping the
// UI served from the engine itself. The cost is UX: the user re-enters the
// key once per page load. Read-only GETs (e.g. /health) stay open
// server-side, but sending the header there too is harmless and keeps one
// code path.
let memoryApiKey = '';

export function getApiKey(): string {
  return memoryApiKey;
}

export function setApiKey(key: string): void {
  memoryApiKey = key || '';
}

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const key = getApiKey();
  const headers = new Headers(init.headers || {});
  if (key && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${key}`);
  }
  return fetch(input, { ...init, headers });
}
