import React, { useState } from 'react';
import { Settings as SettingsIcon, Key, Globe, Terminal, Save } from 'lucide-react';
import { EngineHealth } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: EngineHealth | null;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, health }) => {
  const [apiKey, setApiKey] = useState('');
  const [systemPrompt, setSystemPrompt] = useState(
    'You are an expert low-level systems and MoE streaming inference architect.'
  );
  const [language, setLanguage] = useState('en');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-xl p-6 shadow-2xl flex flex-col space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2.5">
            <SettingsIcon className="w-5 h-5 text-blue-400" />
            <h3 className="text-base font-bold text-slate-100">Engine Settings & Hardware Specs</h3>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 text-sm font-mono px-2 py-1 rounded"
          >
            Esc
          </button>
        </div>

        <div className="space-y-4">
          {/* Memory-only API Key */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Key className="w-3.5 h-3.5 text-amber-400" />
              <span>Memory-Only API Key (Zero Disk Persistence)</span>
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder="sk-live-moe-engine-session..."
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs font-mono text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
          </div>

          {/* System Prompt */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Terminal className="w-3.5 h-3.5 text-indigo-400" />
              <span>Custom System Prompt</span>
            </label>
            <textarea
              rows={3}
              value={systemPrompt}
              onChange={e => setSystemPrompt(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-xl p-3 text-xs text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50 font-sans"
            />
          </div>

          {/* Language Selection */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono flex items-center space-x-2">
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span>Interface Language (i18n)</span>
            </label>
            <select
              value={language}
              onChange={e => setLanguage(e.target.value)}
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

          {/* Hardware Telemetry Card */}
          {health && (
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/60 text-xs font-mono space-y-1.5">
              <div className="text-slate-400">CPU: <span className="text-slate-200">{health.hardware.cpu_brand} ({health.hardware.logical_cores} cores)</span></div>
              <div className="text-slate-400">SIMD: <span className="text-emerald-400">{health.hardware.simd_backend}</span></div>
              <div className="text-slate-400">I/O Engine: <span className="text-cyan-400">{health.hardware.io_backend}</span></div>
              <div className="text-slate-400">System RAM: <span className="text-slate-200">{(health.hardware.total_ram_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB</span></div>
            </div>
          )}
        </div>

        <div className="pt-2 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium"
          >
            Cancel
          </button>
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex items-center space-x-1.5 shadow-lg shadow-blue-600/20"
          >
            <Save className="w-3.5 h-3.5" />
            <span>Apply Settings</span>
          </button>
        </div>
      </div>
    </div>
  );
};
