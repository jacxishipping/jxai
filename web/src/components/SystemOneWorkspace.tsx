import React, { useState } from 'react';
import { Target, CheckCircle2, Loader2, Sparkles } from 'lucide-react';
import { SystemOneResult } from '../types';

export const SystemOneWorkspace: React.FC = () => {
  const [document, setDocument] = useState(
    'The kernel module detected an unexpected page-fault exception in the high-memory I/O driver pool under sustained NVMe streaming load. However, the LFRU fallback buffer caught the fault and routed the request without corruption.'
  );
  const [query, setQuery] = useState('Categorize the system risk severity level:');
  const [optionsStr, setOptionsStr] = useState('Low Risk, Medium Risk, High Risk, Critical Panic');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SystemOneResult | null>(null);

  const handleEvaluate = async () => {
    setIsLoading(true);
    try {
      const opts = optionsStr.split(',').map(s => s.trim()).filter(Boolean);
      const res = await fetch('/v1/systemone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document,
          query,
          options: opts
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
    <div className="flex h-full w-full bg-[#080b13] p-8 overflow-y-auto">
      <div className="max-w-4xl mx-auto w-full flex flex-col space-y-6">
        {/* Header */}
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-2 rounded-xl bg-purple-950/60 border border-purple-800/40 text-purple-400">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-100">System One Decision Engine</h2>
              <p className="text-xs text-slate-400 font-mono">
                Categorical Evaluation with Zero Token Generation Cost (LM Head Projections)
              </p>
            </div>
          </div>
        </div>

        {/* Input Form */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 shadow-xl">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
              Context Document
            </label>
            <textarea
              rows={4}
              value={document}
              onChange={e => setDocument(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 font-sans leading-relaxed"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
                Evaluation Query
              </label>
              <input
                type="text"
                value={query}
                onChange={e => setQuery(e.target.value)}
                className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
                Candidate Options (comma separated)
              </label>
              <input
                type="text"
                value={optionsStr}
                onChange={e => setOptionsStr(e.target.value)}
                className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
              />
            </div>
          </div>

          <div className="pt-2 flex justify-end">
            <button
              onClick={handleEvaluate}
              disabled={isLoading || !document.trim()}
              className="px-6 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 flex items-center space-x-2 shadow-lg shadow-purple-600/20"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Computing Logits...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Score Decisions Instantly</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Scoring Card */}
        {result && (
          <div className="p-6 rounded-2xl bg-slate-900/40 border border-purple-500/30 space-y-5 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <span className="text-xs font-mono text-purple-400 font-semibold uppercase">
                  Top Recommended Decision
                </span>
                <div className="text-2xl font-bold text-white flex items-center space-x-2 mt-1">
                  <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                  <span>{result.recommended}</span>
                </div>
              </div>

              <div className="text-right text-xs font-mono text-slate-400">
                <div>Wall Time: <span className="text-emerald-400 font-bold">{result.wall_ms.toFixed(2)} ms</span></div>
                <div>Tokens Generated: <span className="text-purple-400 font-bold">0 (Pure Math)</span></div>
              </div>
            </div>

            {/* Probability Bars */}
            <div className="space-y-3">
              {result.options.map(opt => {
                const pct = Math.round(opt.probability * 100);
                const isTop = opt.option === result.recommended;
                return (
                  <div key={opt.option} className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className={isTop ? 'text-white font-bold' : 'text-slate-400'}>
                        {opt.option}
                      </span>
                      <span className={isTop ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                        {pct}% (logit: {opt.logit.toFixed(3)})
                      </span>
                    </div>
                    <div className="w-full h-3 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        style={{ width: `${pct}%` }}
                        className={`h-full rounded-full transition-all duration-500 ${
                          isTop
                            ? 'bg-gradient-to-r from-emerald-500 to-cyan-400 shadow-md shadow-emerald-500/30'
                            : 'bg-slate-600'
                        }`}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
