import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Cpu,
  Database,
  Disc,
  ChevronDown,
  ChevronRight,
  Eye,
  Code,
  Zap,
  Clock,
  Layers,
  Play,
  Square,
  Copy,
  Check,
  RotateCcw,
  Pencil,
  Paperclip,
  X,
  Download,
  Trash2,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Loader2
} from 'lucide-react';
import {
  AttachedFile,
  ChatMessage,
  ChatSessionMeta,
  EngineHealth,
  KVSlotInfo,
  LocalModelInfo,
  SamplerSettings
} from '../types';
import { MarkdownRenderer } from '../utils/markdownParser';
import { Translations } from '../utils/i18n';

interface ChatWorkspaceProps {
  health: EngineHealth | null;
  onRefreshHealth?: () => void;
  t: Translations;
  currentSlot: number;
  onSelectSlot: (slotId: number) => void;
  onNewChat: () => void;
  sessions: ChatSessionMeta[];
  setSessions: React.Dispatch<React.SetStateAction<ChatSessionMeta[]>>;
  messagesBySlot: Record<number, ChatMessage[]>;
  setMessagesBySlot: React.Dispatch<React.SetStateAction<Record<number, ChatMessage[]>>>;
  systemPrompt?: string;
  samplerSettings: SamplerSettings;
}

const createWelcomeMessage = (slot: number, health: EngineHealth | null): ChatMessage => ({
  id: 'welcome',
  role: 'assistant',
  content: `### Welcome to the Next-Gen MoE Streaming Inference Engine

Dense attention parameters (GQA/MLA), embedding matrices, and router gating tables stay resident in zero-copy RAM/VRAM, while sparse MoE expert matrices are dynamically streamed from NVMe disk using **PILOT asynchronous overlapped I/O** and **LFRU cache pinning**.

- **Vectorized SIMD & Tensor Core Compute**: AVX-512, AVX2+FMA3, and Vulkan 1.3 Cooperative Matrix.
- **Cache-Aware Routing (CACHE_ROUTE)**: Automatic expert substitution ($\\Delta p < \\epsilon$) to eliminate disk I/O.
- **Interactive Sandbox**: Tag code blocks \`\`\`html or \`\`\`svg to inspect live rendering in the sandbox on the right!`,
  reasoning_content: `MoE Core Initialized. Model: ${health?.model_name || 'Qwen2.5-MoE-Stream-Q4_0'}. All experts mapped with Win32 IOCP / POSIX asynchronous direct I/O prefetch. Dynamic pinning threshold: 12 hits.`,
  telemetry: {
    tok_per_sec: 25.8,
    wall_ms: 38.7,
    io_wait_ms: 0.0,
    cache_hits: health?.tiers.cache_hits ?? 142,
    disk_reads: 0,
    ttft_ms: 12.4,
    queue_latency_ms: 0.8,
    cache_slot: slot
  }
});

export const ChatWorkspace: React.FC<ChatWorkspaceProps> = ({
  health,
  t,
  currentSlot,
  onSelectSlot,
  onNewChat,
  sessions,
  setSessions,
  messagesBySlot,
  setMessagesBySlot,
  systemPrompt,
  samplerSettings,
  onRefreshHealth
}) => {
  const messages = messagesBySlot[currentSlot] || [createWelcomeMessage(currentSlot, health)];

  const updateSlotMessages = (slot: number, updater: (prev: ChatMessage[]) => ChatMessage[]) => {
    setMessagesBySlot(prevMap => {
      const currentList = prevMap[slot] || [createWelcomeMessage(slot, health)];
      return {
        ...prevMap,
        [slot]: updater(currentList)
      };
    });
  };

  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [previewTab, setPreviewTab] = useState<'text' | 'preview'>('preview');
  const [sandboxOverride, setSandboxOverride] = useState<string | null>(null);
  const [collapsedReasoning, setCollapsedReasoning] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [editingMsgId, setEditingMsgId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState('');
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [showSessions, setShowSessions] = useState(true);
  const [models, setModels] = useState<LocalModelInfo[]>([]);
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [slotStats, setSlotStats] = useState<KVSlotInfo | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const fetchModelsAndSlots = async () => {
    try {
      const [mRes, sRes] = await Promise.all([fetch('/v1/models'), fetch('/v1/slots')]);
      if (mRes.ok) {
        const mData = await mRes.json();
        const list = (mData.data || []).filter((m: LocalModelInfo) => m.path);
        setModels(list);
      }
      if (sRes.ok) {
        const sData = await sRes.json();
        const found = (sData.slots || []).find((s: KVSlotInfo) => s.slot_id === currentSlot);
        setSlotStats(found || null);
      }
    } catch {
      // ignore offline preview
    }
  };

  useEffect(() => {
    fetchModelsAndSlots();
  }, [currentSlot, health?.model_name]);

  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === 'function') {
      try {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      } catch {
        // ignore fallback for older browsers or test environments
      }
    }
  }, [messages]);

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  };

  const handleSwitchModel = async (modelPath: string) => {
    if (!modelPath || isSwitchingModel || isStreaming) return;
    setIsSwitchingModel(true);
    try {
      const res = await fetch('/v1/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: modelPath })
      });
      if (res.ok) {
        if (onRefreshHealth) onRefreshHealth();
        await fetchModelsAndSlots();
      }
    } finally {
      setIsSwitchingModel(false);
    }
  };

  const handleClearCurrentSlot = async () => {
    try {
      await fetch('/v1/slots/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot_id: currentSlot })
      });
      await fetchModelsAndSlots();
    } catch {
      // ignore
    }
  };

  const handleFilesSelected = async (fileList: FileList | null) => {
    if (!fileList) return;
    const loaded: AttachedFile[] = [];
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      if (f.size > 512 * 1024) continue; // cap at 512KB per text file
      const text = await f.text();
      loaded.push({ name: f.name, content: text, size: f.size });
    }
    if (loaded.length > 0) {
      setAttachedFiles(prev => [...prev, ...loaded]);
    }
  };

  const streamCompletion = async (
    activeSlot: number,
    historyMessages: { role: 'user' | 'assistant' | 'system'; content: string }[],
    assistantMsgId: string,
    isContinue: boolean
  ) => {
    setIsStreaming(true);
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: false }));

    const requestStartTime = performance.now();
    let ttftRecorded = false;
    let ttftMs = 0;

    try {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          messages: historyMessages,
          stream: true,
          max_tokens: samplerSettings.max_tokens,
          temperature: samplerSettings.temperature,
          top_p: samplerSettings.top_p,
          top_k: samplerSettings.top_k,
          repetition_penalty: samplerSettings.repetition_penalty,
          speculative: samplerSettings.speculative,
          draft_tokens: samplerSettings.draft_tokens,
          cache_slot: activeSlot,
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

            if (delta?.content && !collapsedReasoning[assistantMsgId]) {
              setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: true }));
            }

            updateSlotMessages(activeSlot, prev =>
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
                        cache_slot: activeSlot
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
      if (!(err instanceof Error && err.name === 'AbortError')) {
        updateSlotMessages(activeSlot, prev =>
          prev.map(m =>
            m.id !== assistantMsgId
              ? m
              : {
                  ...m,
                  content: m.content + `\n\n[Connection Error: ${err instanceof Error ? err.message : String(err)}]`
                }
          )
        );
      }
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
      setCollapsedReasoning(prev => ({ ...prev, [assistantMsgId]: true }));
      if (onRefreshHealth) onRefreshHealth();
      fetchModelsAndSlots();
    }
  };

  const handleSend = async () => {
    if ((!input.trim() && attachedFiles.length === 0) || isStreaming) return;
    const isContinue = input.trim().toLowerCase() === 'continue' && attachedFiles.length === 0;
    const activeSlot = currentSlot;

    let finalUserContent = input.trim();
    if (attachedFiles.length > 0) {
      const filesBlock = attachedFiles
        .map(f => `\n\n\`\`\`${f.name.split('.').pop() || 'txt'}\n// File: ${f.name}\n${f.content}\n\`\`\``)
        .join('');
      finalUserContent = (finalUserContent || 'Analyze the attached file(s):') + filesBlock;
    }

    const userMsg: ChatMessage = {
      id: 'msg-' + Date.now(),
      role: 'user',
      content: finalUserContent
    };
    const assistantMsgId = 'asst-' + (Date.now() + 1);
    const initialAssistantMsg: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      reasoning_content: ''
    };

    const historyToSend = [
      ...(systemPrompt ? [{ role: 'system' as const, content: systemPrompt }] : []),
      ...messages
        .filter(m => m.id !== 'welcome' && m.content.trim().length > 0)
        .map(m => ({ role: m.role, content: m.content })),
      { role: 'user' as const, content: userMsg.content }
    ];

    updateSlotMessages(activeSlot, prev => [...prev, userMsg, initialAssistantMsg]);
    setSessions(prev => {
      const exists = prev.some(s => s.slotId === activeSlot);
      const autoTitle = input.trim().slice(0, 36) || attachedFiles[0]?.name || `Session #${activeSlot}`;
      if (!exists) {
        return [{ slotId: activeSlot, title: autoTitle, updatedAt: Date.now() }, ...prev];
      }
      return prev.map(s =>
        s.slotId === activeSlot
          ? {
              ...s,
              title: s.title.startsWith('Session #') ? autoTitle : s.title,
              updatedAt: Date.now()
            }
          : s
      );
    });

    setInput('');
    setAttachedFiles([]);
    setSandboxOverride(null);
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    await streamCompletion(activeSlot, historyToSend, assistantMsgId, isContinue);
  };

  const handleContinue = async (msgId: string) => {
    if (isStreaming) return;
    await streamCompletion(currentSlot, [{ role: 'user', content: 'continue' }], msgId, true);
  };

  const handleRegenerate = async (assistantMsgId: string) => {
    if (isStreaming) return;
    const idx = messages.findIndex(m => m.id === assistantMsgId);
    if (idx <= 0) return;

    const priorMsgs = messages.slice(0, idx);
    const historyToSend = [
      ...(systemPrompt ? [{ role: 'system' as const, content: systemPrompt }] : []),
      ...priorMsgs
        .filter(m => m.id !== 'welcome' && m.content.trim().length > 0)
        .map(m => ({ role: m.role, content: m.content }))
    ];

    updateSlotMessages(currentSlot, prev =>
      prev.slice(0, idx).concat({
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        reasoning_content: ''
      })
    );
    await streamCompletion(currentSlot, historyToSend, assistantMsgId, false);
  };

  const handleSaveEdit = async (userMsgId: string) => {
    if (isStreaming || !editingText.trim()) return;
    const idx = messages.findIndex(m => m.id === userMsgId);
    if (idx < 0) return;

    const editedUserMsg: ChatMessage = { ...messages[idx], content: editingText.trim() };
    const newAsstId = 'asst-' + (Date.now() + 1);
    const newAsstMsg: ChatMessage = {
      id: newAsstId,
      role: 'assistant',
      content: '',
      reasoning_content: ''
    };

    const priorMsgs = messages.slice(0, idx);
    const historyToSend = [
      ...(systemPrompt ? [{ role: 'system' as const, content: systemPrompt }] : []),
      ...priorMsgs
        .filter(m => m.id !== 'welcome' && m.content.trim().length > 0)
        .map(m => ({ role: m.role, content: m.content })),
      { role: 'user' as const, content: editedUserMsg.content }
    ];

    setEditingMsgId(null);
    setEditingText('');
    updateSlotMessages(currentSlot, () => [...priorMsgs, editedUserMsg, newAsstMsg]);
    await streamCompletion(currentSlot, historyToSend, newAsstId, false);
  };

  const handleCopyMessage = (msg: ChatMessage) => {
    navigator.clipboard.writeText(msg.content);
    setCopiedId(msg.id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  const handleExportSession = (format: 'md' | 'json') => {
    const sess = sessions.find(s => s.slotId === currentSlot);
    const title = (sess?.title || `session-${currentSlot}`).replace(/[^a-z0-9_-]+/gi, '_');
    let payload = '';
    let mime = 'text/plain;charset=utf-8';
    if (format === 'json') {
      payload = JSON.stringify({ slot: currentSlot, title: sess?.title, messages }, null, 2);
      mime = 'application/json;charset=utf-8';
    } else {
      payload = messages
        .filter(m => m.id !== 'welcome')
        .map(m => `### ${m.role.toUpperCase()}\n\n${m.content}\n`)
        .join('\n---\n\n');
    }
    const blob = new Blob([payload], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleDeleteSession = (slotId: number) => {
    setSessions(prev => {
      const next = prev.filter(s => s.slotId !== slotId);
      if (next.length === 0) {
        const fallback = { slotId: 0, title: 'Session #0', updatedAt: Date.now() };
        onSelectSlot(0);
        return [fallback];
      }
      if (slotId === currentSlot) {
        onSelectSlot(next[0].slotId);
      }
      return next;
    });
    setMessagesBySlot(prev => {
      const copy = { ...prev };
      delete copy[slotId];
      return copy;
    });
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
  const sandboxHtml =
    sandboxOverride ?? (lastAssistantMessage ? extractCodeForSandbox(lastAssistantMessage.content) : '');

  return (
    <div
      className="flex h-full w-full bg-[#0b0f19]"
      onDragOver={e => e.preventDefault()}
      onDrop={e => {
        e.preventDefault();
        handleFilesSelected(e.dataTransfer.files);
      }}
    >
      {/* Collapsible Session History Sidebar */}
      {showSessions && (
        <div className="w-60 h-full border-r border-slate-800/80 bg-slate-950/60 flex flex-col">
          <div className="h-14 px-4 border-b border-slate-800/80 flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider font-mono text-slate-300">
              Sessions
            </span>
            <div className="flex items-center space-x-1">
              <button
                onClick={onNewChat}
                className="p-1.5 rounded-lg bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 transition-colors"
                title="New Chat Session"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setShowSessions(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
                title="Collapse Sessions"
              >
                <PanelLeftClose className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-2.5 space-y-1">
            {sessions.map(sess => {
              const isActive = sess.slotId === currentSlot;
              return (
                <div
                  key={sess.slotId}
                  onClick={() => onSelectSlot(sess.slotId)}
                  className={`group flex items-center justify-between px-3 py-2 rounded-xl cursor-pointer transition-colors text-xs ${
                    isActive
                      ? 'bg-blue-600/20 border border-blue-500/40 text-slate-100'
                      : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-transparent'
                  }`}
                >
                  <div className="truncate pr-2">
                    <div className="font-medium truncate">{sess.title}</div>
                    <div className="text-[10px] font-mono text-slate-500">KV Slot #{sess.slotId}</div>
                  </div>
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      handleDeleteSession(sess.slotId);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-red-400 transition-opacity"
                    title="Delete session"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              );
            })}
          </div>

          <div className="p-3 border-t border-slate-800/80 flex items-center justify-between space-x-2">
            <button
              onClick={() => handleExportSession('md')}
              className="flex-1 py-1.5 px-2 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300 hover:bg-slate-800 flex items-center justify-center space-x-1"
              title="Export Markdown"
            >
              <Download className="w-3 h-3" />
              <span>.md</span>
            </button>
            <button
              onClick={() => handleExportSession('json')}
              className="flex-1 py-1.5 px-2 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-300 hover:bg-slate-800 flex items-center justify-center space-x-1"
              title="Export JSON"
            >
              <Download className="w-3 h-3" />
              <span>.json</span>
            </button>
          </div>
        </div>
      )}

      {/* Main Chat Stream */}
      <div className="flex flex-col flex-1 h-full min-w-0 border-r border-slate-800/80">
        {/* Top Chat Bar */}
        <div className="h-14 px-4 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/40 backdrop-blur-md">
          <div className="flex items-center space-x-3">
            {!showSessions && (
              <button
                onClick={() => setShowSessions(true)}
                className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
                title="Show Sessions"
              >
                <PanelLeftOpen className="w-4 h-4" />
              </button>
            )}
            <div className="flex items-center space-x-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
              {models.length > 0 ? (
                <div className="flex items-center space-x-2">
                  <select
                    value={models.find(m => m.active)?.path || ''}
                    onChange={e => handleSwitchModel(e.target.value)}
                    disabled={isSwitchingModel || isStreaming}
                    className="bg-slate-900 border border-slate-700/80 rounded-lg px-2.5 py-1 text-xs font-semibold text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    title="Hot-swap local GGUF model"
                  >
                    {models.map(m => (
                      <option key={m.path || m.id} value={m.path || ''}>
                        {m.id} {m.size ? `(${(m.size / (1024 * 1024)).toFixed(0)} MB)` : ''}
                      </option>
                    ))}
                  </select>
                  {isSwitchingModel && <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />}
                </div>
              ) : (
                <span className="font-semibold text-sm text-slate-200">
                  {health?.model_name || t.chatTitle}
                </span>
              )}
            </div>
            <span className="text-xs px-2 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-slate-400 font-mono">
              {health?.architecture || 'qwen2'}
            </span>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono text-slate-400">
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
            <button
              onClick={handleClearCurrentSlot}
              className="flex items-center space-x-1.5 px-2 py-1 rounded-lg bg-slate-900 border border-slate-800 hover:border-emerald-500/40 text-emerald-400 transition-colors"
              title="Click to clear active KV-cache slot"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>
                #{currentSlot}
                {slotStats ? ` (${slotStats.seq_pos}t)` : ''}
              </span>
            </button>
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

                {/* User / Assistant Content */}
                {msg.role === 'user' ? (
                  editingMsgId === msg.id ? (
                    <div className="space-y-2 min-w-[320px]">
                      <textarea
                        rows={3}
                        value={editingText}
                        onChange={e => setEditingText(e.target.value)}
                        className="w-full bg-slate-950/80 border border-slate-600 rounded-xl p-2.5 text-xs text-slate-100 focus:outline-none font-sans"
                      />
                      <div className="flex justify-end space-x-2 text-xs">
                        <button
                          onClick={() => setEditingMsgId(null)}
                          className="px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => handleSaveEdit(msg.id)}
                          className="px-2.5 py-1 rounded-lg bg-emerald-600 text-white hover:bg-emerald-500"
                        >
                          Save & Resubmit
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="text-sm leading-relaxed whitespace-pre-wrap font-sans">
                      {msg.content}
                    </div>
                  )
                ) : (
                  <div className="text-sm leading-relaxed font-sans">
                    {msg.content ? (
                      <MarkdownRenderer
                        content={msg.content}
                        onPreviewInSandbox={code => {
                          setSandboxOverride(code);
                          setPreviewTab('preview');
                        }}
                      />
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

                {/* Message Action Bar: Copy, Edit, Regenerate, Continue */}
                {!isStreaming && msg.id !== 'welcome' && (
                  <div className="mt-2.5 pt-2 border-t border-slate-800/40 flex items-center justify-end space-x-2">
                    <button
                      onClick={() => handleCopyMessage(msg)}
                      className="text-[11px] font-mono text-slate-400 hover:text-slate-200 flex items-center space-x-1 px-2 py-1 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors"
                      title="Copy Markdown"
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                      <span>{copiedId === msg.id ? 'Copied' : 'Copy'}</span>
                    </button>

                    {msg.role === 'user' && (
                      <button
                        onClick={() => {
                          setEditingMsgId(msg.id);
                          setEditingText(msg.content);
                        }}
                        className="text-[11px] font-mono text-slate-200 hover:text-white flex items-center space-x-1 px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
                        title="Edit and resubmit"
                      >
                        <Pencil className="w-3 h-3" />
                        <span>Edit</span>
                      </button>
                    )}

                    {msg.role === 'assistant' && (
                      <>
                        <button
                          onClick={() => handleRegenerate(msg.id)}
                          className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1 px-2 py-1 rounded-lg bg-cyan-950/40 border border-cyan-500/20 hover:bg-cyan-900/40 transition-colors"
                          title="Regenerate response"
                        >
                          <RotateCcw className="w-3 h-3" />
                          <span>Regenerate</span>
                        </button>
                        <button
                          onClick={() => handleContinue(msg.id)}
                          className="text-[11px] font-mono text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 px-2 py-1 rounded-lg bg-indigo-950/40 border border-indigo-500/20 hover:bg-indigo-900/40 transition-colors"
                          title="Continue generating next tokens"
                        >
                          <Play className="w-3 h-3 fill-current" />
                          <span>Continue</span>
                        </button>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Multi-Line Composer + File Attachment Pills */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-900/30 space-y-2.5">
          {attachedFiles.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {attachedFiles.map((file, idx) => (
                <div
                  key={`${file.name}-${idx}`}
                  className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700 text-xs font-mono text-cyan-300"
                >
                  <Paperclip className="w-3 h-3" />
                  <span>{file.name}</span>
                  <span className="text-[10px] text-slate-500">({(file.size / 1024).toFixed(1)}KB)</span>
                  <button
                    onClick={() => setAttachedFiles(prev => prev.filter((_, i) => i !== idx))}
                    className="text-slate-400 hover:text-red-400"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          <form
            onSubmit={e => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-end space-x-2.5"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.md,.cpp,.c,.h,.hpp,.ts,.tsx,.js,.jsx,.py,.json,.csv,.log,.sh,.bat"
              className="hidden"
              onChange={e => {
                handleFilesSelected(e.target.files);
                e.target.value = '';
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isStreaming}
              className="p-3 rounded-xl bg-slate-900/90 border border-slate-700/80 text-slate-400 hover:text-slate-200 hover:border-slate-600 transition-colors disabled:opacity-50"
              title="Attach source file or context (.txt, .md, .cpp, .py, .json, .csv)"
            >
              <Paperclip className="w-4 h-4" />
            </button>

            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={e => {
                setInput(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(180, e.target.scrollHeight)}px`;
              }}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder={`${t.typePrompt} (Shift+Enter for newline)`}
              className="flex-1 bg-slate-900/90 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 font-sans resize-none leading-relaxed max-h-[180px]"
              disabled={isStreaming}
            />

            {isStreaming ? (
              <button
                type="button"
                onClick={handleStop}
                className="px-5 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-medium text-sm transition-all duration-200 shadow-lg shadow-red-600/20 flex items-center space-x-2"
                title="Stop generating"
              >
                <span>Stop</span>
                <Square className="w-3.5 h-3.5 fill-current" />
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() && attachedFiles.length === 0}
                className="px-5 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-600/20 flex items-center space-x-2"
              >
                <span>{t.send}</span>
                <Send className="w-4 h-4" />
              </button>
            )}
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
                srcDoc={'<!DOCTYPE html><html><' + 'head><meta charset="utf-8"><' + 'style>body{margin:0;padding:12px;background:#0f172a;color:#f8fafc;font-family:system-ui;}<' + '/style><' + '/head><body>' + sandboxHtml + '<' + '/body><' + '/html>'}
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
