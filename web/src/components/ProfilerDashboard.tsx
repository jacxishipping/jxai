import React, { useEffect, useState } from 'react';
import { Activity, Zap, HardDrive, Cpu, Clock, Layers, Trash2 } from 'lucide-react';
import { ProfileSample, EngineHealth, SlotsResponse } from '../types';
import { Translations } from '../utils/i18n';
import { apiFetch } from '../utils/api';

interface ProfilerDashboardProps {
  health: EngineHealth | null;
  t: Translations;
}

export const ProfilerDashboard: React.FC<ProfilerDashboardProps> = ({ health, t }) => {
  const [samples, setSamples] = useState<ProfileSample[]>([]);
  const [slotsData, setSlotsData] = useState<SlotsResponse | null>(null);

  const fetchProfile = async () => {
    try {
      const [profRes, slotsRes] = await Promise.all([
        apiFetch('/profile'),
        apiFetch('/v1/slots')
      ]);
      if (profRes.ok) {
        const data = await profRes.json();
        setSamples(data.samples || []);
      }
      if (slotsRes.ok) {
        const sData = await slotsRes.json();
        setSlotsData(sData);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleClearSlot = async (slotId: number) => {
    try {
      await apiFetch('/v1/slots/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slot_id: slotId })
      });
      fetchProfile();
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchProfile();
    const timer = setInterval(fetchProfile, 1500);
    return () => clearInterval(timer);
  }, []);

  const latest = samples[samples.length - 1];
  const maxWall = Math.max(0.5, ...samples.map(s => s.wall_ms));
  const maxTps = Math.max(10, ...samples.map(s => s.tok_per_sec));

  return (
    <div className="flex flex-col h-full w-full bg-[#080b13] p-8 overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-3">
            <Activity className="w-6 h-6 text-emerald-400" />
            <span>{t.profilerTitle}</span>
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-1">
            Breakdown of Matmul vs Attention vs LM Head vs PILOT Disk Reads (sync pread; background prefetch)
          </p>
        </div>

        {latest && (
          <div className="flex items-center space-x-3 bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl">
            <Zap className="w-4 h-4 text-emerald-400 animate-pulse" />
            <div className="text-right font-mono">
              <div className="text-sm font-bold text-emerald-400">{latest.tok_per_sec.toFixed(1)} {t.speed}</div>
              <div className="text-[10px] text-slate-500">Instant Throughput</div>
            </div>
          </div>
        )}
      </div>

      {/* Top 4 Metric Cards */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>Wall-Clock Step</span>
            <Clock className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-slate-100">
            {latest ? `${latest.wall_ms.toFixed(2)} ms` : '--'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1 font-mono">Total forward step latency</div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>I/O Wait</span>
            <HardDrive className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">
            {latest ? `${latest.io_wait_ms.toFixed(2)} ms` : '--'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1 font-mono">Time waiting on expert weight reads</div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>Matmul Time</span>
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-indigo-300">
            {latest ? `${latest.matmul_ms.toFixed(2)} ms` : '--'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1 font-mono">Matmul per forward step</div>
        </div>

        <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 shadow-lg">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
            <span>LFRU Cache Ratio</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-400">
            {health && (health.tiers.cache_hits + health.tiers.cache_misses) > 0
              ? `${Math.round((health.tiers.cache_hits / (health.tiers.cache_hits + health.tiers.cache_misses)) * 100)}%`
              : '--'}
          </div>
          <div className="text-[11px] text-slate-500 mt-1 font-mono">
            {t.hits}: {health?.tiers.cache_hits ?? '—'} | Misses: {health?.tiers.cache_misses ?? '—'}
          </div>
        </div>
      </div>

      {/* Stacked Latency Timeline Chart (I/O Wait vs Matmul vs Attention vs Head) */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 mb-8 flex-1 flex flex-col min-h-[280px]">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider font-mono">
            Rolling Step Latency Breakdown (Stacked ms)
          </h3>
          <div className="flex items-center space-x-4 text-xs font-mono">
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded bg-indigo-500"></span>
              <span className="text-slate-400">Matmul</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded bg-blue-500"></span>
              <span className="text-slate-400">Attention (GQA)</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded bg-purple-500"></span>
              <span className="text-slate-400">LM Head</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded bg-amber-500"></span>
              <span className="text-slate-400">I/O Wait</span>
            </div>
          </div>
        </div>

        <div className="flex-1 flex items-end space-x-2 pt-4 border-b border-slate-800 pb-2">
          {samples.length === 0 ? (
            <div className="w-full text-center text-xs text-slate-500 font-mono py-12">
              Awaiting real forward step telemetry from engine...
            </div>
          ) : (
            samples.map((s, idx) => {
              const hPct = (s.wall_ms / maxWall) * 100;
              const matmulPct = (s.matmul_ms / (s.wall_ms || 1)) * 100;
              const attnPct = (s.attention_ms / (s.wall_ms || 1)) * 100;
              const headPct = (s.lm_head_ms / (s.wall_ms || 1)) * 100;
              const ioPct = (s.io_wait_ms / (s.wall_ms || 1)) * 100;

              return (
                <div key={idx} className="flex-1 flex flex-col items-center group relative h-full justify-end">
                  {/* Tooltip */}
                  <div className="absolute -top-12 bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-[10px] font-mono text-slate-200 opacity-0 group-hover:opacity-100 transition-opacity z-20 pointer-events-none whitespace-nowrap shadow-xl">
                    Wall: {s.wall_ms.toFixed(2)}ms | SIMD: {s.matmul_ms.toFixed(2)}ms | I/O: {s.io_wait_ms.toFixed(2)}ms
                  </div>

                  {/* Stacked Bar */}
                  <div
                    style={{ height: `${Math.max(8, hPct)}%` }}
                    className="w-full rounded-t-lg overflow-hidden flex flex-col-reverse shadow-md"
                  >
                    <div style={{ height: `${matmulPct}%` }} className="bg-indigo-500 w-full" />
                    <div style={{ height: `${attnPct}%` }} className="bg-blue-500 w-full" />
                    <div style={{ height: `${headPct}%` }} className="bg-purple-500 w-full" />
                    <div style={{ height: `${ioPct}%` }} className="bg-amber-500 w-full" />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Throughput Timeline Chart */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 min-h-[220px] flex flex-col mb-8">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider font-mono">
            Decode Throughput Timeline ({t.speed})
          </h3>
          <span className="text-xs font-mono text-emerald-400">Max: {maxTps.toFixed(0)} {t.speed}</span>
        </div>

        <div className="flex-1 flex items-end space-x-2 pt-2 border-b border-slate-800 pb-2">
          {samples.map((s, idx) => {
            const tpsPct = (s.tok_per_sec / maxTps) * 100;
            return (
              <div key={idx} className="flex-1 flex flex-col items-center group relative h-full justify-end">
                <div
                  style={{ height: `${Math.max(5, tpsPct)}%` }}
                  className="w-full bg-gradient-to-t from-emerald-600 to-teal-400 rounded-t-md hover:brightness-125 transition-all"
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* PagedAttention KV-Cache Slot Inspector */}
      <div className="p-6 rounded-2xl bg-slate-900/40 border border-slate-800/80 flex flex-col space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider font-mono">
              PagedAttention KV-Cache Slot Inspector (/v1/slots)
            </h3>
            <p className="text-xs text-slate-500 font-mono mt-0.5">
              {slotsData?.paged_block_manager
                ? `Blocks: ${slotsData.paged_block_manager.used_blocks} / ${slotsData.paged_block_manager.total_blocks} used (${slotsData.paged_block_manager.block_size_tokens} tok/block) • Total KV RAM: ${(slotsData.paged_block_manager.memory_bytes / (1024 * 1024)).toFixed(2)} MB`
                : 'Monitoring active KV-cache slots...'}
            </p>
          </div>
        </div>

        {slotsData && slotsData.slots.length > 0 ? (
          <div className="grid grid-cols-3 gap-4">
            {slotsData.slots.map(slot => (
              <div
                key={slot.slot_id}
                className="p-4 rounded-xl bg-slate-950/70 border border-slate-800 flex flex-col space-y-2 font-mono text-xs"
              >
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="font-bold text-emerald-400">KV Slot #{slot.slot_id}</span>
                  <button
                    onClick={() => handleClearSlot(slot.slot_id)}
                    className="flex items-center space-x-1 px-2 py-0.5 rounded bg-red-950/50 border border-red-500/30 text-red-400 hover:bg-red-900/50 transition-colors text-[11px]"
                    title="Clear KV-cache slot"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Clear</span>
                  </button>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Seq Position:</span>
                  <span className="text-slate-200">{slot.seq_pos} tokens</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Pages Allocated:</span>
                  <span className="text-cyan-400">{slot.pages_allocated} pages</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>KV Memory:</span>
                  <span className="text-indigo-300">{(slot.memory_bytes / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
                <div className="flex justify-between text-slate-400">
                  <span>Compactions:</span>
                  <span className="text-slate-200">{slot.compactions}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-xs font-mono text-slate-500 py-4 text-center">
            No active KV-cache slots reported.
          </div>
        )}
      </div>
    </div>
  );
};
