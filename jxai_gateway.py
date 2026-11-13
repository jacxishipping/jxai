#!/usr/bin/env python3
"""
JXAI API Gateway - OpenAI-compatible proxy to Ollama
Serves as the backend API for Open WebUI
"""
import json
import os
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen, HTTPError
from urllib.error import URLError

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))


class JXAIAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[JXAI Gateway] {args[0]} {args[1]} {args[2]}")

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, User-Agent")

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-JXAI-Engine", "JXAI Inference Engine v2")
        self.end_headers()
        self.wfile.write(body)

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

    def _proxy_stream_to_ollama(self, path, body):
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
            self.send_header("X-JXAI-Engine", "JXAI Inference Engine v2")
            self.end_headers()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
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
                "engine": "JXAI Inference Engine v2",
                "gateway": "JXAI API Gateway",
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
                "service": "JXAI API Gateway",
                "version": "2.0.0",
                "status": "running",
                "message": "JXAI backend API. Open WebUI is the frontend interface."
            })
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        if path in ("/v1/chat/completions", "/v1/completions", "/api/chat", "/api/generate"):
            req_data = json.loads(body.decode("utf-8"))
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
                self._proxy_stream_to_ollama("/api/chat", ollama_body_bytes)
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
    print(f"""
╔══════════════════════════════════════════════╗
║         JXAI API Gateway v2.0                ║
║         OpenAI-compatible proxy to Ollama     ║
╚══════════════════════════════════════════════╝
    """)
    print(f"  Ollama endpoint: {OLLAMA_HOST}")
    print(f"  Gateway port:    {GATEWAY_PORT}")
    print(f"  Listening on:    http://127.0.0.1:{GATEWAY_PORT}")
    print()

    server = HTTPServer(("0.0.0.0", GATEWAY_PORT), JXAIAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down JXAI API Gateway...")
        server.shutdown()
