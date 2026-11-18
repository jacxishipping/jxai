import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, Key, Globe, Terminal, Save, Cpu, Sliders, Zap } from 'lucide-react';
import { EngineHealth, SamplerSettings, SupportedLanguage } from '../types';
import { Translations } from '../utils/i18n';
import { apiFetch, getApiKey, setApiKey } from '../utils/api';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: EngineHealth | null;
  language: SupportedLanguage;
  onSelectLanguage: (lang: SupportedLanguage) => void;
  t: Translations;
  systemPrompt: string;
  onSaveSystemPrompt: (prompt: string) => void;
  samplerSettings: SamplerSettings;
  onSaveSamplerSettings: (settings: SamplerSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  health,
  language,
  onSelectLanguage,
  t,
  systemPrompt,
  onSaveSystemPrompt,
  samplerSettings,
  onSaveSamplerSettings
}) => {
  // API key for the engine (required when MOE_API_KEY is set server-side).
  // Stored in this browser's localStorage as `jxai_api_key` and sent as
  // `Authorization: Bearer <key>` by apiFetch. Clear the field and save
  // to remove it. Never hardcoded.
  const [apiKey, setApiKeyInput] = useState(() => getApiKey());
  const [currentPrompt, setCurrentPrompt] = useState(systemPrompt);
  const [localSampler, setLocalSampler] = useState<SamplerSettings>(samplerSettings);

  useEffect(() => {
    setCurrentPrompt(systemPrompt);
  }, [systemPrompt, isOpen]);

  useEffect(() => {
    setLocalSampler(samplerSettings);
  }, [samplerSettings, isOpen]);

  const presets = [
    {
      name: t.assistantPreset,
      prompt: 'You are an expert AI assistant specializing in high-performance computing, distributed systems, and local MoE streaming architectures.'
    },
    {
      name: t.coderPreset,
      prompt: 'You are an expert systems programmer and compiler engineer. Write clean, idiomatic C++20 and TypeScript code with high performance, rigorous error handling, and zero extraneous dependencies.'
    },
    {
      name: t.concisePreset,
      prompt: 'Provide direct, mathematically precise, and concise answers with zero fluff or conversational filler.'
    },
    {
      name: t.reasonerPreset,
      prompt: 'Think step-by-step through first principles before concluding. Always articulate internal reasoning inside the thinking channel.'
    }
  ];

  if (!isOpen) return null;

  const handleSave = async () => {
    setApiKey(apiKey); // persist (or clear) before the config call below
    onSaveSystemPrompt(currentPrompt);
    onSaveSamplerSettings(localSampler);
    try {
      await apiFetch('/v1/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: localSampler.temperature,
          top_p: localSampler.top_p,
          speculative_draft_tokens: localSampler.draft_tokens,
          max_context: health?.hparams.max_seq_len ?? 32768
        })
      });
    } catch {
      // ignore offline preview
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl p-6 shadow-2xl flex flex-col space-y-6 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2.5">
            <SettingsIcon className="w-5 h-5 text-blue-400" />
            <h3 className="text-base font-bold text-slate-100">{t.settingsTitle}</h3>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-sm font-mono px-2 py-1 rounded"
          >
            Esc
          </button>
        </div>

        <div className="space-y-4">
          {/* API Key (stored in this browser's localStorage) */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Key className="w-3.5 h-3.5 text-amber-400" />
              <span>{t.apiKeyLabel}</span>
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKeyInput(e.target.value)}
              placeholder={t.apiKeyPlaceholder}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs font-mono text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
            <span className="text-[10px] text-slate-500 font-mono mt-1 block">
              Saved in this browser only (localStorage key `jxai_api_key`). Clear the field and save to remove it.
            </span>
          </div>

          {/* System Prompt with Quick Presets */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center space-x-2">
                <Terminal className="w-3.5 h-3.5 text-indigo-400" />
                <span>{t.sysPromptLabel}</span>
              </label>
            </div>

            {/* Presets Row */}
            <div className="flex items-center space-x-1.5 mb-2">
              <span className="text-[11px] font-mono text-slate-400">{t.presets}</span>
              {presets.map(p => (
                <button
                  key={p.name}
                  onClick={() => setCurrentPrompt(p.prompt)}
                  className="px-2 py-0.5 rounded text-[11px] font-mono bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
                >
                  {p.name}
                </button>
              ))}
            </div>

            <textarea
              rows={3}
              value={currentPrompt}
              onChange={e => setCurrentPrompt(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50 font-sans"
            />
          </div>

          {/* Sampling & Speculative Decoding Controls */}
          <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 space-y-3">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="flex items-center space-x-2">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                <span>Sampling & Speculative Decoding</span>
              </span>
              <span className="text-[10px] text-slate-500 font-normal">Synced with /v1/config</span>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs font-mono">
              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Temperature</span>
                  <span className="text-cyan-400">{localSampler.temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={localSampler.temperature}
                  onChange={e => setLocalSampler({ ...localSampler, temperature: Number(e.target.value) })}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Top-P (Nucleus)</span>
                  <span className="text-cyan-400">{localSampler.top_p.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  value={localSampler.top_p}
                  onChange={e => setLocalSampler({ ...localSampler, top_p: Number(e.target.value) })}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Top-K</span>
                  <span className="text-cyan-400">{localSampler.top_k}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={localSampler.top_k}
                  onChange={e => setLocalSampler({ ...localSampler, top_k: Number(e.target.value) })}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Repetition Penalty</span>
                  <span className="text-cyan-400">{localSampler.repetition_penalty.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={1.0}
                  max={1.5}
                  step={0.02}
                  value={localSampler.repetition_penalty}
                  onChange={e => setLocalSampler({ ...localSampler, repetition_penalty: Number(e.target.value) })}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-slate-400 mb-1">
                  <span>Max Tokens</span>
                  <span className="text-cyan-400">{localSampler.max_tokens}</span>
                </div>
                <input
                  type="range"
                  min={64}
                  max={4096}
                  step={64}
                  value={localSampler.max_tokens}
                  onChange={e => setLocalSampler({ ...localSampler, max_tokens: Number(e.target.value) })}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
                />
              </div>

              <div className="flex flex-col justify-between">
                <label className="flex items-center justify-between cursor-pointer text-slate-300">
                  <span className="flex items-center space-x-1.5">
                    <Zap className="w-3.5 h-3.5 text-amber-400" />
                    <span>Speculative MTP</span>
                  </span>
                  <input
                    type="checkbox"
                    checked={localSampler.speculative}
                    onChange={e => setLocalSampler({ ...localSampler, speculative: e.target.checked })}
                    className="accent-amber-500 w-4 h-4 rounded"
                  />
                </label>
                {localSampler.speculative && (
                  <div className="flex items-center justify-between mt-1.5 text-[11px] text-slate-400">
                    <span>Draft Tokens:</span>
                    <input
                      type="number"
                      min={1}
                      max={4}
                      value={localSampler.draft_tokens}
                      onChange={e => setLocalSampler({ ...localSampler, draft_tokens: Math.max(1, Math.min(4, Number(e.target.value))) })}
                      className="w-14 bg-slate-900 border border-slate-700 rounded px-1.5 py-0.5 text-center text-amber-300"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Language Selection */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span>{t.languageLabel}</span>
            </label>
            <select
              value={language}
              onChange={e => onSelectLanguage(e.target.value as SupportedLanguage)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs font-mono text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            >
              <option value="en">English (US)</option>
              <option value="it">Italiano (IT)</option>
              <option value="de">Deutsch (DE)</option>
              <option value="zh-CN">简体中文 (CN)</option>
              <option value="zh-TW">繁體中文 (TW)</option>
              <option value="id">Bahasa Indonesia (ID)</option>
              <option value="pt-BR">Português (BR)</option>
            </select>
          </div>

          {/* Hardware & Tier Residency Telemetry Card */}
          {health && (
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 text-xs font-mono space-y-2">
              <div className="text-slate-300 font-bold flex items-center space-x-1.5 border-b border-slate-800 pb-1.5">
                <Cpu className="w-3.5 h-3.5 text-indigo-400" />
                <span>Hardware & Memory Tier Residency</span>
              </div>
              <div className="text-slate-400">
                CPU: <span className="text-slate-200">{health.hardware.cpu_brand} ({health.hardware.logical_cores} cores)</span>
              </div>
              <div className="text-slate-400">
                SIMD: <span className="text-emerald-400">{health.hardware.simd_backend}</span>
              </div>
              <div className="text-slate-400">
                I/O Engine: <span className="text-cyan-400">{health.hardware.io_backend}</span>
              </div>
              <div className="text-slate-400">
                System RAM: <span className="text-slate-200">{(health.hardware.total_ram_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB (Avail: {(health.hardware.avail_ram_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB)</span>
              </div>
              <div className="text-slate-400">
                Tier Allocation: <span className="text-pink-400">VRAM ({health.tiers.vram_experts})</span> | <span className="text-cyan-400">RAM ({health.tiers.ram_experts})</span> | <span className="text-amber-400">Disk ({health.tiers.disk_experts})</span>
              </div>
            </div>
          )}
        </div>

        <div className="pt-2 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            {t.cancel}
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors shadow-lg shadow-blue-500/20 flex items-center space-x-1.5"
          >
            <Save className="w-3.5 h-3.5" />
            <span>{t.saveSettings}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
