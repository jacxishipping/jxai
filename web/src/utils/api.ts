// Shared fetch wrapper for the engine API.
//
// When the server was started with MOE_API_KEY set, every mutating endpoint
// requires `Authorization: Bearer <key>` (or `x-api-key:`). The key is
// entered in Settings and stored in localStorage under `jxai_api_key` —
// never hardcoded, never shipped in the bundle. Clearing the field in
// Settings removes it. Read-only GETs (e.g. /health) stay open server-side,
// but sending the header there too is harmless and keeps one code path.
export const API_KEY_STORAGE_KEY = 'jxai_api_key';

export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) {
      localStorage.setItem(API_KEY_STORAGE_KEY, key);
    } else {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    }
  } catch {
    // Storage unavailable (private mode, etc.) — key stays memory-only
    // for this page load; requests simply go out without it.
  }
}

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  const key = getApiKey();
  const headers = new Headers(init.headers || {});
  if (key && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${key}`);
  }
  return fetch(input, { ...init, headers });
}
