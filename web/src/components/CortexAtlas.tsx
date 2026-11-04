import React, { useEffect, useRef, useState } from 'react';
import { Activity, Radio, RefreshCw } from 'lucide-react';
import { ExpertTelemetry } from '../types';

export const CortexAtlas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [experts, setExperts] = useState<ExpertTelemetry[]>([]);
  const [selectedExpert, setSelectedExpert] = useState<ExpertTelemetry | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchExperts = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/experts');
      if (res.ok) {
        const data = await res.json();
        setExperts(data.experts || []);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchExperts();
    const timer = setInterval(fetchExperts, 2500);
    return () => clearInterval(timer);
  }, []);

  // 3D Canvas Rendering of Cortical Hemispheres & Expert Nodes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let angle = 0;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const radius = Math.min(cx, cy) * 0.72;

      angle += 0.008;

      // Draw Cortical Outer Ring & Harmonic Grid Lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
      ctx.beginPath();
      ctx.ellipse(cx, cy, radius, radius * 0.45, angle, 0, Math.PI * 2);
      ctx.stroke();

      // Render Expert Nodes
      const total = experts.length || 32;
      experts.forEach((exp, idx) => {
        const theta = (idx / total) * Math.PI * 2 + angle;
        const rLayer = radius * (0.45 + (exp.layer % 4) * 0.14);
        const x = cx + Math.cos(theta) * rLayer;
        const y = cy + Math.sin(theta) * (rLayer * 0.55);

        let fill = '#64748b'; // DISK
        let glow = 'rgba(100, 116, 139, 0.4)';
        if (exp.tier === 'VRAM') {
          fill = '#ec4899'; // VRAM
          glow = 'rgba(236, 72, 153, 0.6)';
        } else if (exp.tier === 'RAM') {
          fill = '#06b6d4'; // RAM
          glow = 'rgba(6, 182, 212, 0.6)';
        }

        const baseSize = 4 + Math.min(10, (exp.activations || 0) * 0.6);
        const isHovered = selectedExpert?.layer === exp.layer && selectedExpert?.expert === exp.expert;

        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, baseSize * (isHovered ? 2.2 : 1.6), 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = isHovered ? '#ffffff' : fill;
        ctx.beginPath();
        ctx.arc(x, y, isHovered ? baseSize + 2 : baseSize, 0, Math.PI * 2);
        ctx.fill();

        if (exp.activations > 0) {
          ctx.strokeStyle = glow;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
      });

      // Center Core
      ctx.fillStyle = 'rgba(99, 102, 241, 0.4)';
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#818cf8';
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [experts, selectedExpert]);

  return (
    <div className="flex h-full w-full bg-[#080c15]">
      {/* 3D Hemisphere Canvas View */}
      <div className="flex-1 flex flex-col items-center justify-center relative p-6">
        <div className="absolute top-6 left-6 flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-indigo-950/60 border border-indigo-800/40 text-indigo-400">
            <Radio className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">3D Cortex Atlas</h2>
            <p className="text-xs text-slate-400 font-mono">Live MoE Semantic Hemisphere & Dynamic Pinning</p>
          </div>
        </div>

        <button
          onClick={fetchExperts}
          disabled={isLoading}
          className="absolute top-6 right-6 p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          title="Refresh Expert Snapshot"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
        </button>

        <canvas
          ref={canvasRef}
          width={640}
          height={520}
          className="rounded-2xl border border-slate-800/60 bg-gradient-to-b from-slate-900/40 to-slate-950/80 shadow-2xl"
        />

        {/* Legend */}
        <div className="mt-4 flex items-center space-x-6 text-xs font-mono">
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 rounded-full bg-pink-500 shadow-lg shadow-pink-500/50"></span>
            <span className="text-slate-300">VRAM Resident (Pinned)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"></span>
            <span className="text-slate-300">RAM LFRU Cache</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3 h-3 rounded-full bg-slate-500"></span>
            <span className="text-slate-400">NVMe Disk Slice</span>
          </div>
        </div>
      </div>

      {/* Expert Matrix Hit-Map & Detail Inspector */}
      <div className="w-[420px] h-full border-l border-slate-800/80 bg-slate-900/30 flex flex-col p-6 overflow-hidden">
        <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider mb-4 flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span>Active Expert Matrix ({experts.length})</span>
        </h3>

        {/* Dense Grid of Expert Chips */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {experts.map(exp => (
            <div
              key={`${exp.layer}-${exp.expert}`}
              onClick={() => setSelectedExpert(exp)}
              className={`p-3 rounded-xl border transition-all cursor-pointer flex items-center justify-between ${
                selectedExpert?.layer === exp.layer && selectedExpert?.expert === exp.expert
                  ? 'border-blue-500 bg-blue-950/40 shadow-lg'
                  : 'border-slate-800/80 bg-slate-900/70 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center space-x-3">
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    exp.tier === 'VRAM' ? 'bg-pink-500 shadow-sm shadow-pink-500' :
                    exp.tier === 'RAM' ? 'bg-cyan-400 shadow-sm shadow-cyan-400' : 'bg-slate-600'
                  }`}
                />
                <div>
                  <div className="text-xs font-semibold text-slate-200">
                    L{exp.layer} : Expert #{exp.expert}
                  </div>
                  <div className="text-[11px] text-slate-400">{exp.topic}</div>
                </div>
              </div>

              <div className="text-right">
                <div className="text-xs font-mono text-cyan-400 font-medium">
                  {exp.activations} hits
                </div>
                <div className="text-[10px] font-mono text-slate-500">
                  LFRU: {exp.lfru_score}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Selected Expert Metadata Card */}
        {selectedExpert && (
          <div className="mt-4 p-4 rounded-xl border border-indigo-500/30 bg-indigo-950/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-indigo-300">Selected Expert Telemetry</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-900/60 text-indigo-200">
                Tier: {selectedExpert.tier}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="text-slate-400">Activations: <span className="text-white font-bold">{selectedExpert.activations}</span></div>
              <div className="text-slate-400">LFRU Score: <span className="text-white font-bold">{selectedExpert.lfru_score}</span></div>
              <div className="text-slate-400">Pinned: <span className="text-white">{selectedExpert.pinned ? 'YES' : 'NO'}</span></div>
              <div className="text-slate-400">Last Prob: <span className="text-cyan-300 font-bold">{selectedExpert.last_prob.toFixed(4)}</span></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
