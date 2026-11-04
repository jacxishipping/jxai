import React, { useState, useRef, useEffect } from 'react';
import { Send, Cpu, Database, Disc, ChevronDown, ChevronRight, Eye, Code, Zap, Clock, Layers, Play } from 'lucide-react';
import { ChatMessage, EngineHealth } from '../types';
import { MarkdownRenderer } from '../utils/markdownParser';
import { Translations } from '../utils/i18n';

interface ChatWorkspaceProps {
  health: EngineHealth | null;
  onRefreshHealth?: () => void;
  t: Translations;
  currentSlot: number;
}

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({ health, t, currentSlot, onRefreshHealth }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `### Welcome to the Next-Gen MoE Streaming Inference Engine

Dense attention parameters (GQA/MLA), embedding matrices, and router gating tables stay resident in zero-copy RAM/VRAM, while sparse MoE expert matrices are dynamically streamed from NVMe disk using **PILOT asynchronous overlapped I/O** and **LFRU cache pinning**.

- **Vectorized SIMD & Tensor Core Compute**: AVX-512, AVX2+FMA3, and Vulkan 1.3 Cooperative Matrix.
- **Cache-Aware Routing (CACHE_ROUTE)**: Automatic expert substitution ($\Delta p < \\epsilon$) to eliminate disk I/O.
- **Interactive Sandbox**: Tag code blocks \`\`\`html or \`\`\`svg to inspect live rendering in the sandbox on the right!`,
      reasoning_content: 'MoE Core Initialized. Model: Qwen2.5-MoE-Stream-Q4_0. All 8 experts per layer mapped to NVMe offsets with Win32 IOCP / POSIX asynchronous direct I/O prefetch. Dynamic pinning threshold: 12 hits.',
      telemetry: {
        tok_per_sec: 2240.5,
        wall_ms: 0.45,
        io_wait_ms: 0.0,
        cache_hits: 142,
        disk_reads: 0,
        ttft_ms: 12.4,
        queue_latency_ms: 0.8,
        cache_slot: currentSlot
      }
    }
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [previewTab, setPreviewTab] = useState<'text' | 'preview'>('preview');
  const [collapsedReasoning, setCollapsedReasoning] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === 'function') {
      try {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      } catch {
        // ignore fallback for older browsers or test environments
      }
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    const isContinue = input.trim().toLowerCase() === 'continue';
    const userMsg: ChatMessage = {
      id: 'msg-' + Date.now(),
      role: 'user',
      content: input
    };
    const assistantMsgId = 'asst-' + (Date.now() + 1);
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning_content: ''
    };

    setMessages(prev => [...prev, userMsg, initialAssistantMsg]);
    setInput('');
    setIsStreaming(true);

    // Auto-expand reasoning accordion while thinking
    setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: false }));

    const requestStartTime = performance.now();
    let ttftRecorded = false;
    let ttftMs = 0;

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: userMsg.content }],
          stream: true,
          max_tokens: 2048,
          temperature: 0.7,
          cache_slot: currentSlot,
          continue_assistant: isContinue,
          context_compaction: true
        })
      });

      if (!response.body) throw new Error('No readable stream');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') break;

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta;
            const telem = parsed.moe_telemetry;

            if (!ttftRecorded && (delta?.reasoning_content || delta?.content)) {
              ttftMs = performance.now() - requestStartTime;
              ttftRecorded = true;
            }

            // Auto-collapse reasoning accordion when content tokens arrive
            if (delta?.content && !collapsedReasoning[assistantMsgId]) {
              setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: true }));
            }

            setMessages(prev =>
              prev.map(m => {
                if (m.id !== assistantMsgId) return m;
                return {
                  ...m,
                  content: delta?.content ? m.content + delta.content : m.content,
                  reasoning_content: delta?.reasoning_content
                    ? (m.reasoning_content || '') + delta.reasoning_content
                    : m.reasoning_content,
                  telemetry: telem
                    ? {
                        tok_per_sec: telem.tok_per_sec,
                        wall_ms: telem.wall_ms,
                        io_wait_ms: telem.io_wait_ms,
                        cache_hits: telem.cache_hits,
                        disk_reads: telem.disk_reads,
                        ttft_ms: ttftMs,
                        queue_latency_ms: Math.max(0.2, (telem.wall_ms || 1) * 0.1),
                        cache_slot: currentSlot
                      }
                    : m.telemetry
                };
              })
            );
          } catch {
            // ignore partial JSON chunk
          }
        }
      }
    } catch (err) {
      setMessages(prev =>
        prev.map(m => {
          if (m.id !== assistantMsgId) return m;
          return {
            ...m,
            content: m.content + `\n\n[Connection Error: ${err instanceof Error ? err.message : String(err)}]`
          };
        })
      );
    } finally {
      setIsStreaming(false);
      setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: true }));
      if (onRefreshHealth) onRefreshHealth();
    }
  };

  const handleContinue = async (msgId: string) => {
    if (isStreaming) return;
    setIsStreaming(true);
    const requestStartTime = performance.now();
    let ttftRecorded = false;
    let ttftMs = 0;

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'continue' }],
          stream: true,
          max_tokens: 2048,
          temperature: 0.7,
          cache_slot: currentSlot,
          continue_assistant: true,
          context_compaction: true
        })
      });

      if (!response.body) throw new Error('No readable stream');
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const dataStr = trimmed.slice(6);
          if (dataStr === '[DONE]') break;

          try {
            const parsed = JSON.parse(dataStr);
            const delta = parsed.choices?.[0]?.delta;
            const telem = parsed.moe_telemetry;

            if (!ttftRecorded && (delta?.reasoning_content || delta?.content)) {
              ttftMs = performance.now() - requestStartTime;
              ttftRecorded = true;
            }

            setMessages(prev =>
              prev.map(m => {
                if (m.id !== msgId) return m;
                return {
                  ...m,
                  content: delta?.content ? m.content + delta.content : m.content,
                  reasoning_content: delta?.reasoning_content
                    ? (m.reasoning_content || '') + delta.reasoning_content
                    : m.reasoning_content,
                  telemetry: telem
                    ? {
                        tok_per_sec: telem.tok_per_sec,
                        wall_ms: telem.wall_ms,
                        io_wait_ms: telem.io_wait_ms,
                        cache_hits: telem.cache_hits,
                        disk_reads: telem.disk_reads,
                        ttft_ms: ttftMs,
                        queue_latency_ms: Math.max(0.2, (telem.wall_ms || 1) * 0.1),
                        cache_slot: currentSlot
                      }
                    : m.telemetry
                };
              })
            );
          } catch {
            // ignore partial JSON chunk
          }
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsStreaming(false);
      if (onRefreshHealth) onRefreshHealth();
    }
  };

  const toggleReasoning = (id: string) => {
    setCollapsedReasoning(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const lastAssistantMessage = messages.filter(m => m.role === 'assistant').slice(-1)[0];
  const extractCodeForSandbox = (text: string) => {
    const match = text.match(/```(?:html|svg|xml)?\s*([\s\S]*?)```/);
    if (match) return match[1];
    if (text.includes('<svg') || text.includes('<div') || text.includes('<canvas')) {
      return text;
    }
    return '';
  };
  const sandboxHtml = lastAssistantMessage ? extractCodeForSandbox(lastAssistantMessage.content) : '';

  return (
    <div className="flex h-full w-full bg-[#0b0f19]">
      {/* Main Chat Stream */}
      <div className="flex flex-col flex-1 h-full min-w-0 border-r border-slate-800/80">
        {/* Top Chat Bar */}
        <div className="h-14 px-6 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/40 backdrop-blur-md">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="font-semibold text-sm text-slate-200">
                {health?.model_name || t.chatTitle}
              </span>
            </div>
            <span className="text-xs px-2 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-slate-400 font-mono">
              {health?.architecture || 'qwen2moe'}
            </span>
          </div>

          <div className="flex items-center space-x-4 text-xs font-mono text-slate-400">
            <div className="flex items-center space-x-1.5" title="VRAM Backbone Residency">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>{t.vram}: {health?.tiers.vram_experts ?? 8}</span>
            </div>
            <div className="flex items-center space-x-1.5" title="LFRU RAM Cache Hits">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              <span>{t.ram}: {health?.tiers.ram_experts ?? 8}</span>
            </div>
            <div className="flex items-center space-x-1.5" title="PILOT Overlapped NVMe Reads">
              <Disc className="w-3.5 h-3.5 text-amber-400" />
              <span>{t.disk}: {health?.tiers.disk_experts ?? 16}</span>
            </div>
            <div className="flex items-center space-x-1.5" title="Active Isolated KV-Cache Slot">
              <Layers className="w-3.5 h-3.5 text-emerald-400" />
              <span>{t.slot}: #{currentSlot}</span>
            </div>
          </div>
        </div>

        {/* Message Feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map(msg => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                  msg.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-indigo-500/10'
                    : 'bg-slate-900/90 border border-slate-800/80 text-slate-200 shadow-xl'
                }`}
              >
                {/* Collapsible Reasoning Accordion (<think> channel) */}
                {msg.reasoning_content && (
                  <div className="mb-4 rounded-xl border border-indigo-500/20 bg-indigo-950/20 overflow-hidden">
                    <button
                      onClick={() => toggleReasoning(msg.id)}
                      className="w-full px-3.5 py-2.5 flex items-center justify-between text-xs font-medium text-indigo-300/90 hover:bg-indigo-900/30 transition-colors"
                    >
                      <div className="flex items-center space-x-2">
                        <Zap className="w-3.5 h-3.5 text-indigo-400" />
                        <span>{t.thinkingChannel}</span>
                      </div>
                      {collapsedReasoning[msg.id] ? (
                        <ChevronRight className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronDown className="w-3.5 h-3.5" />
                      )}
                    </button>
                    {!collapsedReasoning[msg.id] && (
                      <div className="px-3.5 py-2.5 border-t border-indigo-500/10 text-xs font-mono text-indigo-200/70 whitespace-pre-wrap leading-relaxed">
                        {msg.reasoning_content}
                      </div>
                    )}
                  </div>
                )}

                {/* Assistant or User Content with Safe AST Markdown Parser */}
                {msg.role === 'user' ? (
                  <div className="text-sm leading-relaxed whitespace-pre-wrap font-sans">
                    {msg.content}
                  </div>
                ) : (
                  <div className="text-sm leading-relaxed font-sans">
                    {msg.content ? (
                      <MarkdownRenderer content={msg.content} />
                    ) : isStreaming ? (
                      <span className="inline-flex space-x-1 items-center text-slate-400 py-1">
                        <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></span>
                        <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.2s]"></span>
                        <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce [animation-delay:0.4s]"></span>
                      </span>
                    ) : null}
                  </div>
                )}

                {/* Performance & Overlap Metrics Badges */}
                {msg.telemetry && (
                  <div className="mt-3 pt-2.5 border-t border-slate-800/60 flex flex-wrap items-center justify-between text-[11px] font-mono text-slate-400 gap-y-1">
                    <div className="flex items-center space-x-3">
                      <span className="text-emerald-400 font-semibold flex items-center space-x-1">
                        <Zap className="w-3 h-3 text-emerald-400" />
                        <span>{msg.telemetry.tok_per_sec.toFixed(1)} {t.speed}</span>
                      </span>
                      {msg.telemetry.ttft_ms !== undefined && (
                        <span className="text-cyan-400 flex items-center space-x-1">
                          <Clock className="w-3 h-3 text-cyan-400" />
                          <span>{t.ttft}: {msg.telemetry.ttft_ms.toFixed(1)}ms</span>
                        </span>
                      )}
                      <span>Latency: {msg.telemetry.wall_ms.toFixed(1)}ms</span>
                      {msg.telemetry.io_wait_ms > 0 && (
                        <span className="text-amber-400/90">
                          NVMe I/O: {msg.telemetry.io_wait_ms.toFixed(1)}ms
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-cyan-400">{t.hits}: {msg.telemetry.cache_hits}</span>
                      <span className="text-slate-500">|</span>
                      <span className="text-amber-400">{t.reads}: {msg.telemetry.disk_reads}</span>
                    </div>
                  </div>
                )}

                {/* Continue Generation Action Button */}
                {!isStreaming && msg.role === 'assistant' && msg.id !== 'welcome' && (
                  <div className="mt-2.5 pt-2 border-t border-slate-800/40 flex items-center justify-end">
                    <button
                      onClick={() => handleContinue(msg.id)}
                      className="text-[11px] font-mono text-indigo-400 hover:text-indigo-300 flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-indigo-950/40 border border-indigo-500/20 hover:bg-indigo-900/40 transition-colors"
                      title="Continue generating next tokens"
                    >
                      <Play className="w-3 h-3 fill-current" />
                      <span>Continue generation</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Composer */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-900/30">
          <form
            onSubmit={e => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center space-x-3"
          >
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder={t.typePrompt}
              className="flex-1 bg-slate-900/90 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 font-sans"
              disabled={isStreaming}
            />
            <button
              type="submit"
              disabled={isStreaming || !input.trim()}
              className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/20 flex items-center space-x-2"
            >
              <span>{t.send}</span>
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>

      {/* Interactive HTML/SVG Preview Sandbox Panel */}
      <div className="w-96 flex flex-col h-full bg-[#080b13] border-l border-slate-800/80">
        <div className="h-14 px-4 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-sm font-semibold text-slate-200">
            <Code className="w-4 h-4 text-blue-400" />
            <span>{t.interactiveSandbox}</span>
          </div>
          <div className="flex rounded-lg bg-slate-800 p-0.5 border border-slate-700">
            <button
              onClick={() => setPreviewTab('text')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors ${
                previewTab === 'text' ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t.raw}
            </button>
            <button
              onClick={() => setPreviewTab('preview')}
              className={`px-2.5 py-1 text-xs font-medium rounded-md flex items-center space-x-1 transition-colors ${
                previewTab === 'preview' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Eye className="w-3 h-3" />
              <span>{t.preview}</span>
            </button>
          </div>
        </div>

        <div className="flex-1 p-4 overflow-hidden flex flex-col">
          {sandboxHtml ? (
            previewTab === 'preview' ? (
              <iframe
                title="Interactive Sandbox"
                srcDoc={`<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;padding:12px;background:#0f172a;color:#f8fafc;font-family:system-ui;}</style></head><body>${sandboxHtml}</body></html>`}
                sandbox="allow-scripts"
                className="w-full h-full rounded-xl border border-slate-800 bg-slate-900 shadow-inner"
              />
            ) : (
              <pre className="w-full h-full p-3 rounded-xl border border-slate-800 bg-slate-900/90 text-xs font-mono text-cyan-300 overflow-auto whitespace-pre-wrap">
                {sandboxHtml}
              </pre>
            )
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-slate-500">
              <Code className="w-8 h-8 mb-3 opacity-40" />
              <p className="text-xs">{t.noHtmlWidget}</p>
              <p className="text-[11px] text-slate-600 mt-1">
                Prompt the model with "Create an interactive HTML widget" to see live rendering here!
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
