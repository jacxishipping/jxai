import React, { useState, useEffect } from 'react';
import { MessageSquare, Activity, Image as ImageIcon, Target, Settings, Cpu, HardDrive } from 'lucide-react';
import { ChatWorkspace } from './components/ChatWorkspace';
import { CortexAtlas } from './components/CortexAtlas';
import { ProfilerDashboard } from './components/ProfilerDashboard';
import { SystemOneWorkspace } from './components/SystemOneWorkspace';
import { ImageWorkspace } from './components/ImageWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { EngineHealth } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'atlas' | 'profiler' | 'systemone' | 'images'>('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [health, setHealth] = useState<EngineHealth | null>(null);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch {
      // Offline fallback telemetry for client-only previews
      if (!health) {
        setHealth({
          status: 'ok',
          model_name: 'Qwen2.5-MoE-Stream-Q4_0',
          architecture: 'qwen2moe',
          format: 'GGUFv3',
          active_slots: 1,
          queued_requests: 0,
          hparams: {
            vocab_size: 512,
            dim: 128,
            n_layers: 4,
            n_heads: 4,
            n_kv_heads: 2,
            n_experts: 8,
            n_experts_used: 2,
            max_seq_len: 2048
          },
          hardware: {
            cpu_brand: 'x86_64 Multi-Core Processor',
            logical_cores: 8,
            simd_backend: 'AVX2 + FMA3 Vectorized',
            io_backend: 'Win32 IOCP Overlapped Async I/O',
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            avail_ram_bytes: 10 * 1024 * 1024 * 1024,
            vram_budget_bytes: 4 * 1024 * 1024 * 1024
          },
          tiers: {
            vram_experts: 8,
            ram_experts: 8,
            disk_experts: 16,
            total_experts: 32,
            dense_backbone_bytes: 430080,
            sparse_disk_bytes: 884736,
            resident_cache_bytes: 442368,
            cache_hits: 142,
            cache_misses: 12,
            cache_routed: 9,
            bytes_streamed: 331776,
            io_ops: 24
          }
        });
      }
    }
  };

  useEffect(() => {
    fetchHealth();
    const timer = setInterval(fetchHealth, 3000);
    return () => clearInterval(timer);
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setIsSettingsOpen(prev => !prev);
      }
      if (e.key === 'Escape') {
        setIsSettingsOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="flex h-screen w-screen bg-[#070a12] text-slate-100 select-none overflow-hidden">
      {/* Primary Left Navigation Rail */}
      <div className="w-16 h-full flex flex-col items-center justify-between py-5 border-r border-slate-800/80 bg-slate-950/60 z-20">
        <div className="flex flex-col items-center space-y-6">
          {/* Logo */}
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>

          {/* Navigation Tabs */}
          <div className="flex flex-col items-center space-y-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={`p-3 rounded-xl transition-all ${
                activeTab === 'chat'
                  ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30 shadow-md shadow-blue-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="Streaming Chat Workspace"
            >
              <MessageSquare className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('atlas')}
              className={`p-3 rounded-xl transition-all ${
                activeTab === 'atlas'
                  ? 'bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 shadow-md shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="3D Cortex Atlas"
            >
              <Activity className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('profiler')}
              className={`p-3 rounded-xl transition-all ${
                activeTab === 'profiler'
                  ? 'bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 shadow-md shadow-emerald-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="Latency & Tier Profiler"
            >
              <HardDrive className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('systemone')}
              className={`p-3 rounded-xl transition-all ${
                activeTab === 'systemone'
                  ? 'bg-purple-600/20 text-purple-400 border border-purple-500/30 shadow-md shadow-purple-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="System One Decision Engine"
            >
              <Target className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('images')}
              className={`p-3 rounded-xl transition-all ${
                activeTab === 'images'
                  ? 'bg-pink-600/20 text-pink-400 border border-pink-500/30 shadow-md shadow-pink-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
              title="Neural Latent Image Synthesis"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Settings Button */}
        <button
          onClick={() => setIsSettingsOpen(true)}
          className="p-3 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition-all"
          title="Engine Settings (Ctrl+K)"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>

      {/* Main Workspace Area */}
      <div className="flex-1 h-full overflow-hidden">
        {activeTab === 'chat' && <ChatWorkspace health={health} onRefreshHealth={fetchHealth} />}
        {activeTab === 'atlas' && <CortexAtlas />}
        {activeTab === 'profiler' && <ProfilerDashboard health={health} />}
        {activeTab === 'systemone' && <SystemOneWorkspace />}
        {activeTab === 'images' && <ImageWorkspace />}
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        health={health}
      />
    </div>
  );
};
