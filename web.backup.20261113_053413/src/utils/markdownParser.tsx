import React, { useState } from 'react';
import { Copy, Check, Code } from 'lucide-react';

interface CodeBlockProps {
  language: string;
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3 rounded-xl border border-slate-700/80 bg-slate-950 overflow-hidden shadow-md">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-900/90 border-b border-slate-800 text-[11px] font-mono text-slate-400">
        <span className="flex items-center space-x-1.5">
          <Code className="w-3.5 h-3.5 text-blue-400" />
          <span className="uppercase tracking-wider">{language || 'text'}</span>
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center space-x-1 text-slate-400 hover:text-slate-200 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto text-xs font-mono text-slate-200 leading-relaxed whitespace-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
};

// Safe Inline Text Parser for bold, italic, code, strikethrough, links
function parseInlineSpans(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let remaining = text;
  let keyIndex = 0;

  while (remaining.length > 0) {
    // Inline code `code`
    if (remaining.startsWith('`')) {
      const end = remaining.indexOf('`', 1);
      if (end !== -1) {
        nodes.push(
          <code
            key={`code-${keyIndex++}`}
            className="px-1.5 py-0.5 rounded bg-slate-800/80 text-cyan-300 font-mono text-[12px] border border-slate-700/50"
          >
            {remaining.slice(1, end)}
          </code>
        );
        remaining = remaining.slice(end + 1);
        continue;
      }
    }

    // Bold **text**
    if (remaining.startsWith('**')) {
      const end = remaining.indexOf('**', 2);
      if (end !== -1) {
        nodes.push(
          <strong key={`bold-${keyIndex++}`} className="font-semibold text-white">
            {parseInlineSpans(remaining.slice(2, end))}
          </strong>
        );
        remaining = remaining.slice(end + 2);
        continue;
      }
    }

    // Italic *text*
    if (remaining.startsWith('*') && !remaining.startsWith('**')) {
      const end = remaining.indexOf('*', 1);
      if (end !== -1) {
        nodes.push(
          <em key={`italic-${keyIndex++}`} className="italic text-slate-200">
            {parseInlineSpans(remaining.slice(1, end))}
          </em>
        );
        remaining = remaining.slice(end + 1);
        continue;
      }
    }

    // Strikethrough ~~text~~
    if (remaining.startsWith('~~')) {
      const end = remaining.indexOf('~~', 2);
      if (end !== -1) {
        nodes.push(
          <del key={`strike-${keyIndex++}`} className="line-through text-slate-400">
            {parseInlineSpans(remaining.slice(2, end))}
          </del>
        );
        remaining = remaining.slice(end + 2);
        continue;
      }
    }

    // Link [text](url)
    const linkMatch = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch) {
      nodes.push(
        <a
          key={`link-${keyIndex++}`}
          href={linkMatch[2]}
          target="_blank"
          rel="noreferrer noopener"
          className="text-blue-400 underline hover:text-blue-300 transition-colors"
        >
          {linkMatch[1]}
        </a>
      );
      remaining = remaining.slice(linkMatch[0].length);
      continue;
    }

    // Find next special delimiter
    const nextSpecial = remaining.search(/[`*~\[]/);
    if (nextSpecial === -1) {
      nodes.push(remaining);
      break;
    } else if (nextSpecial > 0) {
      nodes.push(remaining.slice(0, nextSpecial));
      remaining = remaining.slice(nextSpecial);
    } else {
      // Single delimiter without match, consume 1 character
      nodes.push(remaining[0]);
      remaining = remaining.slice(1);
    }
  }

  return nodes;
}

export interface MarkdownASTProps {
  content: string;
}

export const MarkdownRenderer: React.FC<MarkdownASTProps> = ({ content }) => {
  if (!content) return null;

  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let i = 0;
  let elemKey = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Fenced Code Block: ```lang
    if (line.trim().startsWith('```')) {
      const lang = line.trim().slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length && lines[i].trim().startsWith('```')) {
        i++; // consume closing ```
      }
      elements.push(
        <CodeBlock key={`block-${elemKey++}`} language={lang} code={codeLines.join('\n')} />
      );
      continue;
    }

    // GFM Table (| Header 1 | Header 2 |)
    if (line.trim().startsWith('|') && line.trim().endsWith('|')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2 && tableLines[1].includes('---')) {
        const headerCells = tableLines[0]
          .slice(1, -1)
          .split('|')
          .map(c => c.trim());
        const bodyRows = tableLines.slice(2).map(r =>
          r
            .slice(1, -1)
            .split('|')
            .map(c => c.trim())
        );

        elements.push(
          <div key={`table-${elemKey++}`} className="my-3 overflow-x-auto rounded-xl border border-slate-800">
            <table className="min-w-full divide-y divide-slate-800 text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-200 font-mono">
                <tr>
                  {headerCells.map((h, hIdx) => (
                    <th key={hIdx} className="px-3.5 py-2 font-semibold">
                      {parseInlineSpans(h)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                {bodyRows.map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-900/40 transition-colors">
                    {row.map((cell, cIdx) => (
                      <td key={cIdx} className="px-3.5 py-2 text-slate-300">
                        {parseInlineSpans(cell)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // Headings: #, ##, ###, ####
    if (line.startsWith('#')) {
      const match = line.match(/^(#{1,4})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const text = match[2];
        if (level === 1) {
          elements.push(
            <h1 key={`h1-${elemKey++}`} className="text-lg font-bold text-slate-100 mt-4 mb-2">
              {parseInlineSpans(text)}
            </h1>
          );
        } else if (level === 2) {
          elements.push(
            <h2 key={`h2-${elemKey++}`} className="text-base font-bold text-slate-200 mt-3 mb-1.5">
              {parseInlineSpans(text)}
            </h2>
          );
        } else if (level === 3) {
          elements.push(
            <h3 key={`h3-${elemKey++}`} className="text-sm font-semibold text-slate-200 mt-2.5 mb-1">
              {parseInlineSpans(text)}
            </h3>
          );
        } else {
          elements.push(
            <h4 key={`h4-${elemKey++}`} className="text-xs font-semibold text-slate-300 mt-2 mb-1">
              {parseInlineSpans(text)}
            </h4>
          );
        }
        i++;
        continue;
      }
    }

    // Blockquote: > text
    if (line.trim().startsWith('>')) {
      const quoteLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('>')) {
        quoteLines.push(lines[i].trim().replace(/^>\s?/, ''));
        i++;
      }
      elements.push(
        <blockquote
          key={`quote-${elemKey++}`}
          className="my-2 border-l-4 border-indigo-500 pl-3.5 py-1 text-xs text-indigo-200/90 italic bg-indigo-950/20 rounded-r-lg"
        >
          {quoteLines.map((ql, qIdx) => (
            <p key={qIdx}>{parseInlineSpans(ql)}</p>
          ))}
        </blockquote>
      );
      continue;
    }

    // Unordered List (- item or * item)
    if (line.trim().match(/^[-*]\s+/)) {
      const listItems: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^[-*]\s+/)) {
        listItems.push(lines[i].trim().replace(/^[-*]\s+/, ''));
        i++;
      }
      elements.push(
        <ul key={`ul-${elemKey++}`} className="my-2 space-y-1 list-disc list-inside text-xs text-slate-300">
          {listItems.map((item, lIdx) => (
            <li key={lIdx} className="leading-relaxed">
              {parseInlineSpans(item)}
            </li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered List (1. item)
    if (line.trim().match(/^\d+\.\s+/)) {
      const listItems: string[] = [];
      while (i < lines.length && lines[i].trim().match(/^\d+\.\s+/)) {
        listItems.push(lines[i].trim().replace(/^\d+\.\s+/, ''));
        i++;
      }
      elements.push(
        <ol key={`ol-${elemKey++}`} className="my-2 space-y-1 list-decimal list-inside text-xs text-slate-300">
          {listItems.map((item, lIdx) => (
            <li key={lIdx} className="leading-relaxed">
              {parseInlineSpans(item)}
            </li>
          ))}
        </ol>
      );
      continue;
    }

    // Normal Paragraph or Line Break
    if (!line.trim()) {
      elements.push(<div key={`br-${elemKey++}`} className="h-2" />);
    } else {
      elements.push(
        <p key={`p-${elemKey++}`} className="text-xs leading-relaxed text-slate-200">
          {parseInlineSpans(line)}
        </p>
      );
    }
    i++;
  }

  return <div className="space-y-1">{elements}</div>;
};

