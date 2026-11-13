import React, { useState } from 'react';
import { Target, CheckCircle2, Loader2, Sparkles, Gauge } from 'lucide-react';
import { SystemOneResult } from '../types';
import { Translations } from '../utils/i18n';

interface SystemOneWorkspaceProps {
  t: Translations;
}

export const SystemOneWorkspace: React.FC<SystemOneWorkspaceProps> = ({ t }) => {
  const [document, setDocument] = useState(
    'The kernel module detected an unexpected page-fault exception in the high-memory I/O driver pool under sustained NVMe streaming load. However, the LFRU fallback buffer caught the fault and routed the request without corruption.'
  );
  const [query, setQuery] = useState('Categorize the system risk severity level:');
  const [optionsStr, setOptionsStr] = useState('Low Risk, Medium Risk, High Risk, Critical Panic');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SystemOneResult | null>(null);

  const calculateEntropyAndConfidence = (
    options: { probability: number }[]
  ): { entropy: number; confidence: 'Very High' | 'High' | 'Moderate' | 'Low' | 'Ambiguous' } => {
    if (options.length <= 1) return { entropy: 0, confidence: 'Very High' };
    let entropy = 0;
    for (const opt of options) {
      if (opt.probability > 1e-9) {
        entropy -= opt.probability * Math.log2(opt.probability);
      }
    }
    const maxEntropy = Math.log2(options.length);
    const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;

    let confidence: 'Very High' | 'High' | 'Moderate' | 'Low' | 'Ambiguous' = 'Moderate';
    if (normalizedEntropy < 0.2) confidence = 'Very High';
    else if (normalizedEntropy < 0.45) confidence = 'High';
    else if (normalizedEntropy < 0.7) confidence = 'Moderate';
    else if (normalizedEntropy < 0.9) confidence = 'Low';
    else confidence = 'Ambiguous';

    return { entropy, confidence };
  };

  const handleEvaluate = async () => {
    setIsLoading(true);
    try {
      const opts = optionsStr
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
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
        const { entropy, confidence } = calculateEntropyAndConfidence(data.options || []);
        setResult({
          ...data,
          entropy,
          confidence
        });
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
              <h2 className="text-xl font-bold text-slate-100">{t.systemOneTitle}</h2>
              <p className="text-xs text-slate-400 font-mono">
                {t.zeroTokensGenerated}
              </p>
            </div>
          </div>
        </div>

        {/* Input Form */}
        <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-4 shadow-xl">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2 font-mono">
              {t.evalDocument}
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
                {t.evalQuery}
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
                {t.candidateOptions}
              </label>
              <input
                type="text"
                value={optionsStr}
                onChange={e => setOptionsStr(e.target.value)}
                className="w-full bg-slate-950/80 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-purple-500/50 font-mono"
              />
            </div>
          </div>

          <div className="pt-2 flex justify-end">
            <button
              onClick={handleEvaluate}
              disabled={isLoading || !document.trim()}
              className="px-6 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium text-sm transition-all duration-200 disabled:opacity-50 flex items-center space-x-2 shadow-lg shadow-purple-600/20 cursor-pointer"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Evaluating LM Head Logits...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>{t.evaluate}</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results & Shannon Entropy Visualization */}
        {result && (
          <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 space-y-6 shadow-xl">
            {/* Top Recommended Banner */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div className="flex items-center space-x-2.5">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                <span className="text-sm font-semibold text-slate-200">Recommended Selection:</span>
                <span className="text-base font-bold text-white px-3 py-0.5 rounded-lg bg-emerald-950/80 border border-emerald-500/40 text-emerald-300">
                  {result.recommended}
                </span>
              </div>
              <div className="text-xs font-mono text-slate-400">
                Evaluation Step: <span className="text-purple-400 font-bold">{result.wall_ms.toFixed(3)} ms</span> (0 Tokens Generated)
              </div>
            </div>

            {/* Entropy & Confidence Gauges */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center space-x-3">
                <Gauge className="w-5 h-5 text-cyan-400" />
                <div>
                  <div className="text-[11px] font-mono text-slate-400">{t.entropy} (H)</div>
                  <div className="text-base font-bold font-mono text-cyan-300">
                    {result.entropy !== undefined ? `${result.entropy.toFixed(3)} bits` : '--'}
                  </div>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 flex items-center space-x-3">
                <Sparkles className="w-5 h-5 text-purple-400" />
                <div>
                  <div className="text-[11px] font-mono text-slate-400">{t.confidence}</div>
                  <div className="text-base font-bold font-mono text-purple-300">
                    {result.confidence || 'High'}
                  </div>
                </div>
              </div>
            </div>

            {/* Probability Bars */}
            <div className="space-y-3">
              {result.options.map((opt, idx) => {
                const pct = Math.round(opt.probability * 100);
                const isTop = idx === 0;
                return (
                  <div key={opt.option} className="space-y-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className={isTop ? 'text-white font-bold' : 'text-slate-300'}>
                        {opt.option} {isTop && '★'}
                      </span>
                      <span className="text-purple-400 font-bold">
                        {pct}% ({opt.probability.toFixed(4)})
                      </span>
                    </div>
                    <div className="w-full bg-slate-950 rounded-full h-3 overflow-hidden p-0.5 border border-slate-800">
                      <div
                        style={{ width: `${Math.max(2, pct)}%` }}
                        className={`h-full rounded-full transition-all duration-500 ${
                          isTop
                            ? 'bg-gradient-to-r from-purple-500 to-indigo-500'
                            : 'bg-slate-700'
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
