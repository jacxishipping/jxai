import React, { useState, useEffect } from 'react';
import { Image as ImageIcon, Sparkles, Maximize2, X, ZoomIn, ZoomOut, Loader2, Download, Info } from 'lucide-react';
import { ImageResult } from '../types';
import { Translations } from '../utils/i18n';
import { apiFetch } from '../utils/api';

interface ImageWorkspaceProps {
  t: Translations;
}

export const ImageWorkspace: React.FC<ImageWorkspaceProps> = ({ t }) => {
  const [prompt, setPrompt] = useState('Cybernetic cortical MoE neural network with pulsing harmonic light streams');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [steps, setSteps] = useState(12);
  const [guidance, setGuidance] = useState(7.5);
  const [seed, setSeed] = useState(42);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImageResult | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showMetadata, setShowMetadata] = useState(true);

  const resolutionPresets = [
    { label: '256x256', w: 256, h: 256 },
    { label: '512x512', w: 512, h: 512 },
    { label: '768x768', w: 768, h: 768 },
    { label: '1024x1024', w: 1024, h: 1024 },
    { label: '1280x768', w: 1280, h: 768 },
    { label: '2048x1024', w: 2048, h: 1024 }
  ];

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await apiFetch('/v1/images/generations', {
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
      } else {
        let detail = `Server returned ${res.status}.`;
        try {
          const errBody = await res.json();
          if (errBody?.message) detail = errBody.message;
        } catch {
          // keep the status fallback
        }
        setError(detail);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed — is the engine running?');
    } finally {
      setIsLoading(false);
    }
  };

  // Close the lightbox with Escape.
  useEffect(() => {
    if (!lightboxImage) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setLightboxImage(null);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [lightboxImage]);

  const handleExportPng = (dataUri: string) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0, width, height);
        const pngUrl = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `moe-synthesis-seed${seed}-${width}x${height}.png`;
        link.href = pngUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    };
    img.src = dataUri;
  };

  return (
    <div className="flex h-full w-full bg-[#080b13] overflow-hidden flex-col lg:flex-row">
      {/* Settings Sidebar */}
      <div className="w-full lg:w-80 h-full border-r border-slate-800/80 bg-slate-900/40 p-6 flex flex-col space-y-5 overflow-y-auto shrink-0">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-pink-950/60 border border-pink-800/40 text-pink-400">
            <ImageIcon className="w-5 h-5" />
          </div>
          <div>
            {/* Not image generation: the engine renders a 16x16 latent SVG
                grid conditioned by a transformer forward pass. The i18n
                keys imageTitle / generateImage / denoisingLatents still say
                "image" / "synthesis" — they are intentionally not used here. */}
            <h2 className="text-sm font-bold text-slate-100">Latent Visualisation</h2>
            <p className="text-[11px] text-slate-400 font-mono">SVG grid conditioned by MoE logits — not a generated image</p>
          </div>
        </div>

        <div>
          <label htmlFor="jxai-visual-prompt" className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
            {t.prompt}
          </label>
          <textarea
            id="jxai-visual-prompt"
            rows={3}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl p-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-pink-500/50"
          />
        </div>

        {/* Resolution Presets & Custom Multiples of 32 */}
        <div className="space-y-4 pt-1">
          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1.5">
              <span>{t.resolution}</span>
              <span className="text-pink-400 font-bold">{width}x{height}</span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 mb-2">
              {resolutionPresets.map(preset => (
                <button
                  key={preset.label}
                  aria-pressed={width === preset.w && height === preset.h}
                  aria-label={`Resolution ${preset.label}`}
                  onClick={() => {
                    setWidth(preset.w);
                    setHeight(preset.h);
                  }}
                  className={`py-1 text-[11px] font-mono rounded-lg border transition-all ${
                    width === preset.w && height === preset.h
                      ? 'bg-pink-950/60 border-pink-500 text-pink-300 shadow-sm shadow-pink-500/20'
                      : 'bg-slate-950/50 border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">
              Aligned to SIMD tile multiples of 32 (range 256 to 2048)
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>{t.eulerSteps}</span>
              <span className="text-pink-400">{steps}</span>
            </div>
            <input
              type="range"
              aria-label="Visualisation steps"
              min={4}
              max={32}
              step={1}
              value={steps}
              onChange={e => setSteps(Number(e.target.value))}
              className="w-full accent-pink-500 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-400 mb-1">
              <span>{t.cfgScale}</span>
              <span className="text-pink-400">{guidance.toFixed(1)}</span>
            </div>
            <input
              type="range"
              aria-label="Guidance scale"
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
              <span>{t.seed}</span>
              <span className="text-pink-400 font-mono">#{seed}</span>
            </div>
            <input
              type="number"
              aria-label="Seed"
              value={seed}
              onChange={e => setSeed(Number(e.target.value))}
              className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3 py-1.5 text-xs font-mono text-slate-100"
            />
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isLoading || !prompt.trim()}
          className="w-full py-3 rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 flex items-center justify-center space-x-2 shadow-lg shadow-pink-600/20 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-pink-400"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Rendering visualisation…</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Render Visualisation</span>
            </>
          )}
        </button>
        {error && (
          <div role="alert" className="rounded-xl border border-red-500/40 bg-red-950/40 px-3.5 py-2.5 text-xs text-red-300">
            Visualisation failed: {error}
          </div>
        )}
      </div>

      {/* Main Canvas & Progressive Step Previews */}
      <div className="flex-1 h-full p-8 overflow-y-auto flex flex-col items-center justify-center">
        {result ? (
          <div className="flex flex-col items-center space-y-6 max-w-3xl w-full">
            {/* Final Render with Lightbox Zoom Trigger */}
            <div className="relative group rounded-2xl overflow-hidden border border-slate-800 shadow-2xl bg-slate-950">
              <img
                src={result.data[0].url}
                alt="Latent visualisation (SVG grid) for the given prompt"
                className="w-full max-w-[480px] aspect-square h-auto object-cover"
              />
              <div className="absolute top-4 right-4 flex items-center space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => handleExportPng(result.data[0].url)}
                  aria-label={t.exportPng}
                  className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-700 text-white hover:bg-slate-900 transition-colors shadow-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-pink-400"
                  title={t.exportPng}
                >
                  <Download className="w-4 h-4" />
                </button>
                <button
                  onClick={() => {
                    setLightboxImage(result.data[0].url);
                    setZoomLevel(1);
                  }}
                  aria-label="Open lightbox"
                  className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-700 text-white hover:bg-slate-900 transition-colors shadow-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-pink-400"
                  title="Open Lightbox"
                >
                  <Maximize2 className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Latent preview steps — illustrative SVG grids, not a diffusion render */}
            <div className="w-full bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5">
              <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 font-mono flex items-center justify-between">
                <span>Latent Preview Steps (illustrative)</span>
                <span className="text-pink-400 font-normal">Completed in {result.elapsed_ms.toFixed(1)} ms</span>
              </div>
              <div className="grid grid-cols-4 gap-4">
                {result.previews.map((prev, idx) => (
                  <div key={idx} className="flex flex-col items-center space-y-1.5">
                    <img
                      src={prev.url}
                      alt={`Latent preview step ${prev.step}`}
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
            <p className="text-sm">No visualisation rendered yet.</p>
            <p className="text-xs text-slate-600">Configure parameters on the left and click Render Visualisation.</p>
          </div>
        )}
      </div>

      {/* Interactive Modal Lightbox with Pan / Zoom up to 3x & PNG Export */}
      {lightboxImage && (
        <div className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-8">
          <div className="relative max-w-4xl max-h-[90vh] flex flex-col items-center">
            {/* Toolbar */}
            <div className="absolute top-4 right-4 flex items-center space-x-2 z-10">
              <button
                onClick={() => setShowMetadata(prev => !prev)}
                aria-label="Toggle metadata"
                className={`p-2 rounded-lg border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
                  showMetadata ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:bg-slate-800'
                }`}
                title="Toggle Metadata"
              >
                <Info className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoomLevel(prev => Math.min(3, +(prev + 0.5).toFixed(1)))}
                aria-label={t.zoomIn}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
                title={t.zoomIn}
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoomLevel(prev => Math.max(1, +(prev - 0.5).toFixed(1)))}
                aria-label={t.zoomOut}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
                title={t.zoomOut}
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleExportPng(lightboxImage)}
                aria-label={t.exportPng}
                className="p-2 rounded-lg bg-emerald-700/80 border border-emerald-600 text-white hover:bg-emerald-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
                title={t.exportPng}
              >
                <Download className="w-4 h-4" />
              </button>
              <button
                onClick={() => setLightboxImage(null)}
                aria-label={t.close}
                className="p-2 rounded-lg bg-slate-900/80 border border-slate-700 text-white hover:bg-red-900/80 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
                title={t.close}
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950 p-2">
              <img
                src={lightboxImage}
                alt="Latent visualisation, zoomed"
                style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.2s ease-out' }}
                className="max-h-[75vh] max-w-full object-contain cursor-grab"
              />
            </div>

            {/* Metadata Bar */}
            {showMetadata && result && (
              <div className="mt-3 w-full bg-slate-900/90 border border-slate-800 rounded-xl px-4 py-2.5 text-xs font-mono text-slate-300 flex items-center justify-between">
                <div>
                  Resolution: <span className="text-pink-400">{width}x{height}</span> | Seed: <span className="text-pink-400">#{seed}</span> | Steps: <span className="text-pink-400">{steps}</span> | CFG: <span className="text-pink-400">{guidance.toFixed(1)}</span>
                </div>
                <div className="text-slate-400">
                  Zoom: <span className="text-indigo-400">{zoomLevel.toFixed(1)}x</span> (Max 3.0x)
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
