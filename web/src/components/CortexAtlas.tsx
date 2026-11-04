import React, { useEffect, useRef, useState } from 'react';
import { Activity, RefreshCw, RotateCw, ZoomIn, ZoomOut, Zap } from 'lucide-react';
import { ExpertTelemetry } from '../types';
import { Translations } from '../utils/i18n';

interface CortexAtlasProps {
  t: Translations;
}

export const CortexAtlas: React.FC<CortexAtlasProps> = ({ t }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [experts, setExperts] = useState<ExpertTelemetry[]>([]);
  const [selectedExpert, setSelectedExpert] = useState<ExpertTelemetry | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [yaw, setYaw] = useState(0.4);
  const [pitch, setPitch] = useState(0.2);
  const [zoom, setZoom] = useState(1.0);
  const [isLiveRouting, setIsLiveRouting] = useState(true);
  const isDraggingRef = useRef(false);
  const lastMousePosRef = useRef({ x: 0, y: 0 });

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

  // Mouse drag handlers for 3D Camera Rotation (Yaw & Pitch)
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDraggingRef.current = true;
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingRef.current) return;
    const dx = e.clientX - lastMousePosRef.current.x;
    const dy = e.clientY - lastMousePosRef.current.y;
    lastMousePosRef.current = { x: e.clientX, y: e.clientY };

    setYaw(prev => prev + dx * 0.01);
    setPitch(prev => Math.max(-1.2, Math.min(1.2, prev + dy * 0.01)));
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  // 3D Canvas Rendering of Cortical Hemispheres & Expert Nodes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let localYaw = yaw;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      const baseRadius = Math.min(cx, cy) * 0.65 * zoom;

      if (!isDraggingRef.current) {
        localYaw += 0.004; // subtle ambient rotation
      } else {
        localYaw = yaw;
      }

      // Draw 3D Cortical Grid Rings with Yaw & Pitch Rotation
      const numRings = 7;
      for (let r = 0; r < numRings; r++) {
        const ringFrac = (r + 1) / numRings;
        const ringRad = baseRadius * ringFrac;

        ctx.strokeStyle = `rgba(56, 189, 248, ${0.05 + ringFrac * 0.1})`;
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.ellipse(
          cx,
          cy,
          ringRad,
          ringRad * Math.abs(Math.cos(pitch) * 0.6 + 0.4),
          localYaw,
          0,
          Math.PI * 2
        );
        ctx.stroke();
      }

      // Render 3D MoE Expert Nodes
      const total = experts.length || 32;
      experts.forEach((exp, idx) => {
        // Distribute in 3D cortical spherical mantle
        const phi = (idx / total) * Math.PI * 2;
        const mantleRadius = baseRadius * (0.55 + (exp.layer % 4) * 0.12);

        // 3D coordinates on hemisphere
        const X0 = Math.cos(phi) * mantleRadius;
        const Y0 = ((exp.expert % 4) - 1.5) * (mantleRadius * 0.35);
        const Z0 = Math.sin(phi) * mantleRadius;

        // Apply Yaw (Y-axis rotation)
        const X1 = X0 * Math.cos(localYaw) - Z0 * Math.sin(localYaw);
        const Z1 = X0 * Math.sin(localYaw) + Z0 * Math.cos(localYaw);

        // Apply Pitch (X-axis rotation)
        const Y2 = Y0 * Math.cos(pitch) - Z1 * Math.sin(pitch);
        const Z2 = Y0 * Math.sin(pitch) + Z1 * Math.cos(pitch);

        // Perspective Projection
        const pFactor = 1.0 / (1.0 + Z2 / (baseRadius * 2.8));
        const px = cx + X1 * pFactor;
        const py = cy + Y2 * pFactor;

        let fill = '#64748b'; // DISK
        let glow = 'rgba(100, 116, 139, 0.4)';
        if (exp.tier === 'VRAM') {
          fill = '#ec4899'; // VRAM
          glow = 'rgba(236, 72, 153, 0.7)';
        } else if (exp.tier === 'RAM') {
          fill = '#06b6d4'; // RAM
          glow = 'rgba(6, 182, 212, 0.7)';
        }

        const isHovered = selectedExpert?.layer === exp.layer && selectedExpert?.expert === exp.expert;
        const isFired = isLiveRouting && exp.activations > 0;
        const baseSize = (4 + Math.min(10, (exp.activations || 0) * 0.5)) * pFactor;

        // Fired / Active Flashing
        if (isFired) {
          ctx.strokeStyle = glow;
          ctx.lineWidth = 1.2 * pFactor;
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(px, py);
          ctx.stroke();

          ctx.fillStyle = glow;
          ctx.beginPath();
          ctx.arc(px, py, baseSize * 2.0, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.fillStyle = isHovered ? '#ffffff' : fill;
        ctx.beginPath();
        ctx.arc(px, py, isHovered ? baseSize * 1.5 : baseSize, 0, Math.PI * 2);
        ctx.fill();

        // Node Label
        if (isHovered || baseSize > 8) {
          ctx.font = '10px monospace';
          ctx.fillStyle = '#f8fafc';
          ctx.fillText(`L${exp.layer}:E${exp.expert}`, px + 6, py - 4);
        }
      });

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [experts, yaw, pitch, zoom, selectedExpert, isLiveRouting]);

  return (
    <div className="flex h-full w-full bg-[#080b13] overflow-hidden">
      {/* 3D Interactive Canvas */}
      <div className="flex-1 h-full relative flex flex-col">
        {/* Header Bar */}
        <div className="h-14 px-6 border-b border-slate-800/80 flex items-center justify-between bg-slate-900/40 backdrop-blur-md z-10">
          <div className="flex items-center space-x-3">
            <Activity className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-sm font-bold text-slate-100">{t.atlasTitle}</h2>
              <p className="text-[11px] text-slate-400 font-mono">
                Interactive 3D Cortical Mantle & Sparse MoE Expert Routing
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Live Routing Mode Toggle */}
            <button
              onClick={() => setIsLiveRouting(prev => !prev)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors ${
                isLiveRouting
                  ? 'bg-cyan-950/60 border-cyan-500 text-cyan-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5 text-cyan-400" />
              <span>Live Routing Mode</span>
            </button>

            {/* Camera Controls */}
            <div className="flex items-center space-x-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5">
              <button
                onClick={() => setZoom(z => Math.min(2.0, +(z + 0.1).toFixed(1)))}
                className="p-1.5 text-slate-400 hover:text-slate-200"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setZoom(z => Math.max(0.5, +(z - 0.1).toFixed(1)))}
                className="p-1.5 text-slate-400 hover:text-slate-200"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => { setYaw(0.4); setPitch(0.2); setZoom(1.0); }}
                className="p-1.5 text-slate-400 hover:text-slate-200"
                title="Reset Camera"
              >
                <RotateCw className="w-3.5 h-3.5" />
              </button>
            </div>

            <button
              onClick={fetchExperts}
              disabled={isLoading}
              className="p-2 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200"
              title="Refresh Telemetry"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* 3D Canvas with Mouse Drag Interaction */}
        <div className="flex-1 relative cursor-grab active:cursor-grabbing">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            className="w-full h-full block"
          />
          <div className="absolute bottom-4 left-6 text-[11px] font-mono text-slate-500 select-none">
            Click & drag to rotate 3D yaw/pitch • Scroll or use buttons to zoom
          </div>
        </div>
      </div>

      {/* Expert Inspection Sidebar */}
      <div className="w-80 h-full border-l border-slate-800/80 bg-slate-900/40 p-6 flex flex-col space-y-4 overflow-y-auto">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider font-mono">
          MoE Expert Matrix ({experts.length} Nodes)
        </h3>

        {/* Expert List Grid */}
        <div className="grid grid-cols-4 gap-2">
          {experts.map(exp => (
            <button
              key={`e-${exp.layer}-${exp.expert}`}
              onClick={() => setSelectedExpert(exp)}
              className={`p-2 rounded-xl border text-center transition-all ${
                selectedExpert?.layer === exp.layer && selectedExpert?.expert === exp.expert
                  ? 'border-white bg-slate-800 shadow-lg'
                  : exp.tier === 'VRAM'
                  ? 'border-pink-500/40 bg-pink-950/20 text-pink-300'
                  : exp.tier === 'RAM'
                  ? 'border-cyan-500/40 bg-cyan-950/20 text-cyan-300'
                  : 'border-slate-800 bg-slate-950/40 text-slate-400'
              }`}
            >
              <div className="text-[10px] font-mono text-slate-400">L{exp.layer}</div>
              <div className="text-xs font-bold font-mono">E{exp.expert}</div>
              <div className="text-[9px] font-mono mt-0.5 opacity-80">{exp.tier}</div>
            </button>
          ))}
        </div>

        {/* Selected Expert Deep Telemetry */}
        {selectedExpert && (
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3 font-mono text-xs">
            <div className="text-slate-200 font-bold border-b border-slate-800 pb-2 flex justify-between items-center">
              <span>Layer {selectedExpert.layer} • Expert {selectedExpert.expert}</span>
              <span
                className={`px-2 py-0.5 rounded text-[10px] ${
                  selectedExpert.tier === 'VRAM'
                    ? 'bg-pink-900 text-pink-200'
                    : selectedExpert.tier === 'RAM'
                    ? 'bg-cyan-900 text-cyan-200'
                    : 'bg-slate-800 text-slate-300'
                }`}
              >
                {selectedExpert.tier}
              </span>
            </div>
            <div>
              <span className="text-slate-500">Semantic Topic:</span>
              <div className="text-cyan-300 font-sans text-xs mt-0.5">{selectedExpert.topic}</div>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">LFRU Cache Score:</span>
              <span className="text-slate-200">{selectedExpert.lfru_score}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Total Forward Hits:</span>
              <span className="text-slate-200">{selectedExpert.activations}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Dynamic Pinning:</span>
              <span className={selectedExpert.pinned ? 'text-emerald-400 font-bold' : 'text-slate-400'}>
                {selectedExpert.pinned ? 'Pinned (Hot)' : 'Dynamic LFRU'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
