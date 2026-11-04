import React, { useState } from 'react';
import { Image as ImageIcon, Sparkles, Maximize2, X, ZoomIn, ZoomOut, Loader2 } from 'lucide-react';
import { ImageResult } from '../types';

export const ImageWorkspace: React.FC = () => {
  const [prompt, setPrompt] = useState('Cybernetic cortical MoE neural network with pulsing harmonic light streams');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(12);
  const [guidance, setGuidance] = useState(7.5);
  const [seed, setSeed] = useState(42);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ImageResult | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/v1/images/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          width,
          height,
          steps,
          guidance_scale: guidance,
          seed
        })
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-full w-full bg-[#080b13] overflow-hidden">
      {/* Settings Sidebar */}
      <div className="w-80 h-full border-r border-slate-800/80 bg-slate-900/40 p-6 flex flex-col space-y-5 overflow-y-auto">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-pink-950/60 border border-pink-800/40 text-pink-400">
            <ImageIcon className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-100">Neural Latent Synthesis</h2>
            <p className="text-[11px] text-slate-400 font-mono">Denoising Conditioned by MoE Logits</p>
          </div>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
            Prompt
          </label>
          <textarea
            rows={3}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl p-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-pink-500/50"
          />
        </div>

        {/* Sliders */}
        <div className="space-y-4 pt-2">
          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>Resolution</span>
              <span className="text-pink-400">{width}x{height}</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[256, 512, 768].map(sz => (
                <button
                  key={sz}
                  onClick={() => { setWidth(sz); setHeight(sz); }}
                  className={`py-1.5 text-xs font-mono rounded-lg border transition-all ${
                    width === sz
                      ? 'bg-pink-950/50 border-pink-500 text-pink-300'
                      : 'bg-slate-950/50 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {sz}x{sz}
                </button>
              ))}
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>Euler Denoise Steps</span>
              <span className="text-pink-400">{steps}</span>
            </div>
            <input
              type="range"
              min={4}
              max={24}
              value={steps}
              onChange={e => setSteps(Number(e.target.value))}
              className="w-full accent-pink-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>Guidance Scale (CFG)</span>
              <span className="text-pink-400">{guidance.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={15.0}
              step={0.5}
              value={guidance}
              onChange={e => setGuidance(Number(e.target.value))}
              className="w-full accent-pink-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>Random Seed</span>
              <span className="text-pink-400">#{seed}</span>
            </div>
            <input
              type="number"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs font-mono text-slate-100"
            />
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isLoading || !prompt.trim()}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 flex items-center justify-center space-x-2 shadow-lg shadow-pink-600/20"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Denoising Latents...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Generate Image</span>
            </>
          )}
        </button>
      </div>

      {/* Main Canvas & Progressive Step Previews */}
      <div className="flex-1 h-full p-8 overflow-y-auto flex flex-col items-center justify-center">
        {result ? (
          <div className="flex flex-col items-center space-y-6 max-w-3xl w-full">
            {/* Final Render with Lightbox Zoom Trigger */}
            <div className="relative group rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950">
              <img
                src={result.data[0].url}
                alt="Generated synthesis"
                className="w-[480px] h-[480px] object-cover"
              />
              <button
                onClick={() => { setLightboxImage(result.data[0].url); setZoomLevel(1); }}
                className="absolute top-4 right-4 p-2.5 rounded-xl bg-slate-950/80 border border-slate-700/80 text-white opacity-0 group-hover:opacity-100 transition-opacity shadow-lg"
                title="Open Lightbox"
              >
                <Maximize2 className="w-4 h-4" />
              </button>
            </div>

            {/* Progressive Denoising Step Previews Timeline */}
            <div className="w-full bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5">
              <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 font-mono flex items-center justify-between">
                <span>Euler-Maruyama Progressive Denoising Preview</span>
                <span className="text-pink-400 font-normal">Completed in {result.elapsed_ms.toFixed(1)} ms</span>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {result.previews.map((prev, idx) => (
                  <div key={idx} className="flex flex-col items-center space-y-1.5">
                    <img
                      src={prev.url}
                      alt={`Step ${prev.step}`}
                      className="w-full h-24 object-cover rounded-xl border border-slate-800"
                    />
                    <span className="text-[10px] font-mono text-slate-400">
                      Step {prev.step}/{prev.total_steps} (σ={prev.sigma.toFixed(2)})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center text-slate-500 space-y-3">
            <ImageIcon className="w-12 h-12 opacity-30" />
            <p className="text-sm">No image synthesized yet.</p>
            <p className="text-xs text-slate-600">Configure parameters on the left and click Generate.</p>
          </div>
        )}
      </div>

      {/* Interactive Modal Lightbox with Pan / Zoom */}
      {lightboxImage && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-8">
          <div className="relative max-w-4xl max-h-[90vh] flex flex-col items-center">
            {/* Toolbar */}
            <div className="absolute top-4 right-4 flex items-center space-x-2 z-10">
              <button
                onClick={() => setZoomLevel(prev => Math.min(3, prev + 0.5))}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoomLevel(prev => Math.max(1, prev - 0.5))}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => setLightboxImage(null)}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-red-900/80"
                title="Close"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 p-2">
              <img
                src={lightboxImage}
                alt="Modal zoom"
                style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.2s ease-out' }}
                className="max-h-[80vh] max-w-full object-contain cursor-grab"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
