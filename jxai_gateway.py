#!/usr/bin/env python3
"""
JXAI Ollama Proxy — OPTIONAL integration, NOT the native engine.

This is a legacy OpenAI-compatible proxy to Ollama. It exists only for users
who explicitly want to run inference through Ollama + Open WebUI. It is NOT
part of the default JXAI path: the default backend is the native C++ engine
(`moe_engine serve` on 127.0.0.1:8001, started by start_jxai.sh), which the
project's own web UI talks to directly. Do not mistake this proxy for the
native engine — its /health reports "ollama_connected", and every chat
request is forwarded to the Ollama daemon on 127.0.0.1:11434.

Run it only deliberately, e.g.:
    JXAI_ALLOW_OLLAMA_PROXY=1 python3 jxai_gateway.py --port 8001

Security (2026-11-17): binds 127.0.0.1 by default (port 8001). When
MOE_API_KEY is set, every POST endpoint requires
`Authorization: Bearer <key>` or `x-api-key: <key>` (401 otherwise);
GET /health stays open as the liveness probe. A non-loopback bind with
no key configured is refused unless --allow-insecure-remote is passed.
"""
import argparse
import hmac
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen, HTTPError
from urllib.error import URLError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8001"))
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
API_KEY = os.environ.get("MOE_API_KEY", "")
# Comma-separated exact origins (same contract as native moe_engine /
# MOE_CORS_ORIGINS). Empty = same-origin only: no Access-Control-Allow-Origin.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("MOE_CORS_ORIGINS", "").split(",") if o.strip()
]

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_authorized(handler):
    """True when no key is configured, or the request carries the right one."""
    if not API_KEY:
        return True
    auth = handler.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], API_KEY):
        return True
    xkey = handler.headers.get("x-api-key", "")
    if xkey and hmac.compare_digest(xkey, API_KEY):
        return True
    return False


class JXAIAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[JXAI Gateway] {args[0]} {args[1]} {args[2]}")

    def _set_cors(self):
        # Mirror native engine: never emit Access-Control-Allow-Origin: *.
        # When MOE_CORS_ORIGINS is set, echo the request Origin only if it
        # is an exact allow-list match; otherwise omit CORS headers entirely.
        origin = self.headers.get("Origin", "")
        if origin and origin in CORS_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
            self.send_header("Access-Control-Allow-Headers",
                             "Content-Type, Authorization, x-api-key, User-Agent")

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-JXAI-Engine", "JXAI Ollama Proxy (optional; NOT the native engine)")
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self):
        """401 + False when a configured key is missing/wrong on this request."""
        if is_authorized(self):
            return True
        self._send_json(401, {
            "error": "unauthorized",
            "message": "Missing or invalid API key. Send Authorization: Bearer <key> or x-api-key: <key>.",
        })
        return False

    def _proxy_to_ollama(self, path, body=None, method="GET"):
        url = f"{OLLAMA_HOST}{path}"
        req = Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            resp = urlopen(req)
            return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            return e.code, {"error": str(e)}
        except URLError as e:
            return 503, {"error": f"Ollama not reachable: {str(e)}"}

    def _proxy_stream_to_ollama(self, path, body, openai_format=True, model_name="qwen3.6:35b-a3b"):
        url = f"{OLLAMA_HOST}{path}"
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            resp = urlopen(req)
            self.send_response(200)
            self._set_cors()
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-JXAI-Engine", "JXAI Ollama Proxy (optional; NOT the native engine)")
            self.end_headers()
            cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_ts = int(time.time())
            while True:
                line = resp.readline()
                if not line:
                    break
                if not openai_format:
                    self.wfile.write(line)
                    self.wfile.flush()
                    continue
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                try:
                    item = json.loads(line_str)
                    msg = item.get("message", {})
                    content = msg.get("content", "")
                    done = item.get("done", False)
                    sse_obj = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": item.get("model", model_name),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content} if content else {},
                                "finish_reason": "stop" if done else None,
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(sse_obj)}\n\n".encode("utf-8"))
                    if done:
                        self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            self._send_json(e.code, {"error": error_body})
        except URLError as e:
            self._send_json(503, {"error": f"Ollama not reachable: {str(e)}"})

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self):
        path = self.path

        if path == "/health":
            ollama_ok = False
            try:
                req = Request(f"{OLLAMA_HOST}/api/tags")
                resp = urlopen(req, timeout=2)
                ollama_ok = resp.status == 200
            except Exception:
                pass
            self._send_json(200, {
                "status": "ok",
                "engine": "JXAI Ollama Proxy (optional; NOT the native engine)",
                "gateway": "JXAI Ollama Proxy (optional)",
                "ollama_connected": ollama_ok,
                "version": "2.0.0"
            })
            return

        if path == "/v1/models" or path == "/api/tags":
            status, data = self._proxy_to_ollama("/api/tags")
            if status == 200:
                models = data.get("models", [])
                openai_models = {
                    "object": "list",
                    "data": [
                        {
                            "id": m["name"],
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "jxai"
                        }
                        for m in models
                    ]
                }
                self._send_json(200, openai_models)
            else:
                self._send_json(status, data)
            return

        if path == "/":
            self._send_json(200, {
                "service": "JXAI Ollama Proxy (optional; NOT the native engine)",
                "version": "2.0.0",
                "status": "running",
                "message": "Optional Ollama proxy. The default JXAI backend is the native C++ engine (moe_engine serve on 127.0.0.1:8001)."
            })
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path
        # Mutating endpoints require the API key whenever one is configured.
        if not self._require_auth():
            return
        # Same 1 MiB body cap as the native engine (DoS bound).
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            self._send_json(400, {"error": "invalid_content_length",
                                  "message": "Content-Length must be a non-negative integer."})
            return
        if content_length < 0 or content_length > 1 * 1024 * 1024:
            self._send_json(413, {"error": "request_too_large",
                                  "message": "Request body exceeds the 1 MiB limit."})
            return
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        if path in ("/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"):
            try:
                req_data = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid_json",
                                      "message": "Request body must be valid JSON."})
                return
            stream = req_data.get("stream", False)
            ollama_model = req_data.get("model", "qwen3.6:35b-a3b")
            messages = req_data.get("messages", [])

            ollama_body = {
                "model": ollama_model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": req_data.get("temperature", 0.7),
                    "top_p": req_data.get("top_p", 0.9),
                    "top_k": req_data.get("top_k", 40),
                }
            }
            if "max_tokens" in req_data:
                ollama_body["options"]["num_predict"] = req_data["max_tokens"]

            ollama_body_bytes = json.dumps(ollama_body).encode("utf-8")

            if stream:
                is_openai = path.startswith("/v1/")
                self._proxy_stream_to_ollama("/api/chat", ollama_body_bytes, openai_format=is_openai, model_name=ollama_model)
            else:
                status, data = self._proxy_to_ollama("/api/chat", ollama_body_bytes, "POST")
                if status == 200:
                    openai_resp = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": ollama_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": data.get("message", {}).get("content", "")
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        }
                    }
                    self._send_json(200, openai_resp)
                else:
                    self._send_json(status, data)
            return

        self._send_json(404, {"error": "Not found"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JXAI Ollama Proxy (OPTIONAL; not the native engine)")
    parser.add_argument("--host", default=GATEWAY_HOST,
                        help="Bind host (default: 127.0.0.1; env GATEWAY_HOST)")
    parser.add_argument("--port", type=int, default=GATEWAY_PORT,
                        help="Bind port (default: 8001; env GATEWAY_PORT)")
    parser.add_argument("--allow-insecure-remote", action="store_true",
                        help="Allow a non-loopback bind with no MOE_API_KEY configured")
    args = parser.parse_args()

    # Require an explicit opt-in so this proxy cannot be mistaken for the
    # native engine by a casual `python3 jxai_gateway.py`.
    if os.environ.get("JXAI_ALLOW_OLLAMA_PROXY", "") != "1":
        print("""
[ERROR] Refusing to start the optional Ollama proxy without an explicit opt-in.

  export JXAI_ALLOW_OLLAMA_PROXY=1
  python3 jxai_gateway.py --port 8001

This is NOT the native JXAI engine. The default backend is:
  ./start_jxai.sh   # moe_engine serve on 127.0.0.1:8001
""", file=sys.stderr)
        sys.exit(1)

    if args.host not in LOOPBACK_HOSTS and not API_KEY and not args.allow_insecure_remote:
        print(f"""
[SECURITY] Refusing to bind {args.host} without an API key: set MOE_API_KEY,
bind 127.0.0.1, or pass --allow-insecure-remote to accept the risk explicitly.
""", file=sys.stderr)
        sys.exit(1)
    if args.host not in LOOPBACK_HOSTS and not API_KEY:
        print(f"""
[SECURITY WARNING] Binding {args.host} with NO API key
(--allow-insecure-remote). Anyone who can reach this port can use the
models. This is strongly discouraged.
""", file=sys.stderr)
    elif args.host not in LOOPBACK_HOSTS:
        print(f"[SECURITY] Binding {args.host} — API-key auth is enabled.")
    elif not API_KEY:
        print("[SECURITY] Listening on loopback only with no API key configured. "
              "Set MOE_API_KEY to require auth.")

    print(f"""
╔══════════════════════════════════════════════╗
║  JXAI Ollama Proxy v2.0 (OPTIONAL)           ║
║  NOT the native engine — proxy to Ollama     ║
╚══════════════════════════════════════════════╝
    """)
    print("[WARNING] This is the OPTIONAL Ollama proxy. The default JXAI backend is")
    print("          the native C++ engine (moe_engine serve on 127.0.0.1:8001).")
    print(f"  Ollama endpoint: {OLLAMA_HOST}")
    print(f"  Gateway port:    {args.port}")
    print(f"  Listening on:    http://{args.host}:{args.port}")
    print(f"  Auth:            {'API key required (MOE_API_KEY)' if API_KEY else 'disabled (loopback only, no key set)'}")
    print()

    server = HTTPServer((args.host, args.port), JXAIAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down JXAI Ollama Proxy...")
        server.shutdown()
