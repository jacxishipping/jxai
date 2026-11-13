import React, { useState, useEffect } from 'react';
import {
  MessageSquare,
  Activity,
  Image as ImageIcon,
  Target,
  Settings,
  Cpu,
  Search,
  Plus,
  Sun,
  Moon
} from 'lucide-react';
import { ChatWorkspace } from './components/ChatWorkspace';
import { CortexAtlas } from './components/CortexAtlas';
import { ProfilerDashboard } from './components/ProfilerDashboard';
import { SystemOneWorkspace } from './components/SystemOneWorkspace';
import { ImageWorkspace } from './components/ImageWorkspace';
import { SettingsModal } from './components/SettingsModal';
import { EngineHealth, SupportedLanguage } from './types';
import { translations } from './utils/i18n';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'atlas' | 'profiler' | 'systemone' | 'images'>('chat');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [language, setLanguage] = useState<SupportedLanguage>('en');
  const [systemPrompt, setSystemPrompt] = useState(
    'You are an expert AI assistant specializing in high-performance computing and local MoE streaming architectures.'
  );
  const [currentSlot, setCurrentSlot] = useState(0);
  const [health, setHealth] = useState<EngineHealth | null>(null);

  const t = translations[language] || translations.en;

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        setHealth(data);
      }
    } catch {
      // Offline fallback telemetry for client previews
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
      // Ctrl/Cmd + K: Open conversation search dialog
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsSearchOpen(prev => !prev);
      }
      // Ctrl/Cmd + N: Create new chat (advances slot)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'n') {
        e.preventDefault();
        setCurrentSlot(prev => prev + 1);
        setActiveTab('chat');
      }
      // Ctrl/Cmd + Shift + L: Toggle Light / Dark theme
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'l') {
        e.preventDefault();
        setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
      }
      // Esc: Close open modal/dialogs
      if (e.key === 'Escape') {
        setIsSettingsOpen(false);
        setIsSearchOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const searchResults = [
    { id: '1', title: 'Explain MoE streaming architecture', tab: 'chat' as const },
    { id: '2', title: 'Interactive 3D Cortex Atlas', tab: 'atlas' as const },
    { id: '3', title: 'Latency Profiler & Overlapped I/O', tab: 'profiler' as const },
    { id: '4', title: 'System One Decision Engine', tab: 'systemone' as const },
    { id: '5', title: 'Neural Latent Image Synthesis', tab: 'images' as const }
  ].filter(item => item.title.toLowerCase().includes(searchQuery.toLowerCase()));

  return (
    <div
      className={`flex h-screen w-screen select-none overflow-hidden ${
        theme === 'dark' ? 'bg-[#070a12] text-slate-100' : 'bg-slate-100 text-slate-900'
      }`}
    >
      {/* Primary Left Navigation Rail */}
      <div
        className={`w-16 h-full flex flex-col items-center justify-between py-5 border-r z-20 ${
          theme === 'dark'
            ? 'border-slate-800/80 bg-slate-950/60'
            : 'border-slate-300 bg-white shadow-sm'
        }`}
      >
        <div className="flex flex-col items-center space-y-6">
          {/* Logo */}
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>

          {/* New Chat Button */}
          <button
            onClick={() => {
              setCurrentSlot(prev => prev + 1);
              setActiveTab('chat');
            }}
            className="w-10 h-10 rounded-xl bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 flex items-center justify-center transition-all border border-blue-500/30"
            title={`${t.newChat} (Ctrl+N) - Slot #${currentSlot + 1}`}
          >
            <Plus className="w-5 h-5" />
          </button>

          {/* Quick Search */}
          <button
            onClick={() => setIsSearchOpen(true)}
            className="w-10 h-10 rounded-xl bg-slate-800/40 hover:bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center justify-center transition-all"
            title={`${t.searchDialogTitle} (Ctrl+K)`}
          >
            <Search className="w-4 h-4" />
          </button>

          {/* Workspaces */}
          <div className="flex flex-col items-center space-y-2">
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activeTab === 'chat'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/30'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
              title={t.chatTitle}
            >
              <MessageSquare className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('atlas')}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activeTab === 'atlas'
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/30'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
              title={t.atlasTitle}
            >
              <Activity className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('profiler')}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activeTab === 'profiler'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/30'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
              title={t.profilerTitle}
            >
              <Cpu className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('systemone')}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activeTab === 'systemone'
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
              title={t.systemOneTitle}
            >
              <Target className="w-5 h-5" />
            </button>

            <button
              onClick={() => setActiveTab('images')}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 ${
                activeTab === 'images'
                  ? 'bg-pink-600 text-white shadow-lg shadow-pink-600/30'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
              title={t.imageTitle}
            >
              <ImageIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Bottom Actions: Theme Toggle & Settings */}
        <div className="flex flex-col items-center space-y-3">
          <button
            onClick={() => setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))}
            className="w-10 h-10 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-slate-200 flex items-center justify-center transition-colors"
            title={`${t.toggleTheme} (Ctrl+Shift+L)`}
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-indigo-500" />}
          </button>

          <button
            onClick={() => setIsSettingsOpen(true)}
            className="w-10 h-10 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-slate-200 flex items-center justify-center transition-colors"
            title={t.settingsTitle}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 h-full overflow-hidden relative">
        {activeTab === 'chat' && (
          <ChatWorkspace health={health} t={t} currentSlot={currentSlot} onRefreshHealth={fetchHealth} />
        )}
        {activeTab === 'atlas' && <CortexAtlas t={t} />}
        {activeTab === 'profiler' && <ProfilerDashboard health={health} t={t} />}
        {activeTab === 'systemone' && <SystemOneWorkspace t={t} />}
        {activeTab === 'images' && <ImageWorkspace t={t} />}
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        health={health}
        language={language}
        onSelectLanguage={setLanguage}
        t={t}
        systemPrompt={systemPrompt}
        onSaveSystemPrompt={setSystemPrompt}
      />

      {/* Ctrl+K Conversation Search Dialog */}
      {isSearchOpen && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-start justify-center pt-24 p-6">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg p-5 shadow-2xl space-y-4">
            <div className="flex items-center space-x-3 border-b border-slate-800 pb-3">
              <Search className="w-5 h-5 text-blue-400" />
              <input
                type="text"
                autoFocus
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder={t.searchPlaceholder}
                className="w-full bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none font-sans"
              />
              <span className="text-xs font-mono text-slate-500 px-1.5 py-0.5 rounded border border-slate-800">
                Esc
              </span>
            </div>

            <div className="space-y-1.5 max-h-64 overflow-y-auto">
              {searchResults.length > 0 ? (
                searchResults.map(item => (
                  <button
                    key={item.id}
                    onClick={() => {
                      setActiveTab(item.tab);
                      setIsSearchOpen(false);
                    }}
                    className="w-full px-3.5 py-2.5 rounded-xl text-left text-xs font-mono text-slate-300 hover:bg-slate-800 hover:text-white transition-colors flex items-center justify-between"
                  >
                    <span>{item.title}</span>
                    <span className="text-[10px] uppercase text-slate-500">{item.tab}</span>
                  </button>
                ))
              ) : (
                <div className="text-center py-6 text-xs text-slate-500 font-mono">
                  No matching sessions or commands found.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
