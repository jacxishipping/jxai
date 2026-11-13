import React, { useState } from 'react';
import { Settings as SettingsIcon, Key, Globe, Terminal, Save, Cpu } from 'lucide-react';
import { EngineHealth, SupportedLanguage } from '../types';
import { Translations } from '../utils/i18n';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: EngineHealth | null;
  language: SupportedLanguage;
  onSelectLanguage: (lang: SupportedLanguage) => void;
  t: Translations;
  systemPrompt: string;
  onSaveSystemPrompt: (prompt: string) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  health,
  language,
  onSelectLanguage,
  t,
  systemPrompt,
  onSaveSystemPrompt
}) => {
  // Memory-only API key: NOT written to localStorage or sessionStorage!
  const [apiKey, setApiKey] = useState('');
  const [currentPrompt, setCurrentPrompt] = useState(systemPrompt);

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

  const handleSave = () => {
    onSaveSystemPrompt(currentPrompt);
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
          {/* Memory-only API Key (Zero Disk Persistence) */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Key className="w-3.5 h-3.5 text-amber-400" />
              <span>{t.apiKeyLabel}</span>
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={t.apiKeyPlaceholder}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs font-mono text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
            <span className="text-[10px] text-slate-500 font-mono mt-1 block">
              Purged from disk and local storage to prevent key leakage.
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
