'use client';

import { useState } from 'react';
import type { SearchIndex, SearchIndexSegment } from '@/lib/types';

function fmtTs(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Renders `text` with every occurrence of `query` highlighted.
 *
 * Server-side extractors mark emphasis with **double asterisks**; those are
 * unwrapped and highlighted too, so raw markdown never leaks into the UI.
 *
 * Replaces an earlier implementation that rendered the full text and then
 * appended the matched terms a second time, so a search for "budget" showed
 * "...the budget today" followed by a stray "budget".
 */
function Highlighted({ text, query }: { text: string; query: string }) {
  const q = query.trim();
  // Split on markdown emphasis first so it can be styled and stripped.
  const emphasisParts = text.split(/(\*\*[^*]+\*\*)/g);

  return (
    <>
      {emphasisParts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
          return (
            <mark key={i} className="rounded bg-amber-100 px-0.5 text-amber-900">
              {part.slice(2, -2)}
            </mark>
          );
        }
        if (!q) return <span key={i}>{part}</span>;

        const chunks = part.split(new RegExp(`(${escapeRegExp(q)})`, 'gi'));
        return (
          <span key={i}>
            {chunks.map((chunk, j) =>
              chunk.toLowerCase() === q.toLowerCase() ? (
                <mark key={j} className="rounded bg-amber-100 px-0.5 text-amber-900">
                  {chunk}
                </mark>
              ) : (
                chunk
              ),
            )}
          </span>
        );
      })}
    </>
  );
}

interface Props {
  index: SearchIndex;
}

const MAX_TEXT_RESULTS = 50;

export function SearchBox({ index }: Props) {
  const [query, setQuery] = useState('');

  const q = query.trim().toLowerCase();

  // Preset filters. These used to be undiscoverable magic strings buried in the
  // placeholder — they are now clickable chips that fill the same query.
  const presets = [
    { label: 'Action items', query: 'action items', count: index.action_items?.length ?? 0 },
    { label: 'Decisions', query: 'decisions', count: index.decisions?.length ?? 0 },
    { label: 'Questions', query: 'questions', count: index.questions?.length ?? 0 },
  ].filter((p) => p.count > 0);

  let results: Array<{ ts: number; speaker: string; text: string; type?: string }> = [];
  let truncated = false;

  if (q === 'action items' && index.action_items?.length) {
    results = index.action_items.map((a) => ({
      ts: a.timestamp,
      speaker: index.speakers?.[a.speaker_id] ?? a.speaker_id,
      text: a.text,
      type: 'Action Item',
    }));
  } else if (q === 'decisions' && index.decisions?.length) {
    results = index.decisions.map((d) => ({
      ts: d.timestamp,
      speaker: index.speakers?.[d.speaker_id] ?? d.speaker_id,
      text: d.text,
      type: 'Decision',
    }));
  } else if (q === 'questions' && index.questions?.length) {
    results = index.questions.map((qn) => ({
      ts: qn.timestamp,
      speaker: index.speakers?.[qn.speaker_id] ?? qn.speaker_id,
      text: qn.text,
      type: 'Question',
    }));
  } else if (q.length >= 2) {
    const matches = (index.segments ?? []).filter((seg: SearchIndexSegment) =>
      seg.text.toLowerCase().includes(q),
    );
    truncated = matches.length > MAX_TEXT_RESULTS;
    results = matches
      .slice(0, MAX_TEXT_RESULTS)
      .map((seg: SearchIndexSegment) => ({
        ts: seg.start,
        speaker: seg.speaker_display ?? index.speakers?.[seg.speaker_id] ?? seg.speaker_id,
        text: seg.text,
      }));
  }

  const isPresetQuery = presets.some((p) => p.query === q);
  const showResults = q.length >= 2;

  return (
    <div>
      <div className="relative">
        <svg
          aria-hidden="true"
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="search"
          aria-label="Search transcript"
          className="w-full rounded-lg border border-slate-300 py-2.5 pl-9 pr-9 text-sm transition-colors placeholder:text-slate-400 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          placeholder="Search the transcript…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            type="button"
            onClick={() => setQuery('')}
            aria-label="Clear search"
            className="absolute right-2 top-1/2 -translate-y-1/2 cursor-pointer rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {presets.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-xs text-slate-400">Jump to:</span>
          {presets.map((p) => {
            const active = q === p.query;
            return (
              <button
                key={p.query}
                type="button"
                onClick={() => setQuery(active ? '' : p.query)}
                aria-pressed={active}
                className={`inline-flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                  active
                    ? 'border-indigo-200 bg-indigo-50 text-indigo-700'
                    : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                {p.label}
                <span className={active ? 'text-indigo-500' : 'text-slate-400'}>{p.count}</span>
              </button>
            );
          })}
        </div>
      )}

      {showResults && (
        <>
          <p className="mt-4 text-xs text-slate-500" aria-live="polite">
            {results.length === 0
              ? 'No matches'
              : `${results.length}${truncated ? '+' : ''} ${results.length === 1 ? 'match' : 'matches'}`}
            {truncated && (
              <span className="text-slate-400">
                {' '}· showing the first {MAX_TEXT_RESULTS}, refine your search to narrow it down
              </span>
            )}
          </p>

          <div className="mt-2 max-h-[520px] space-y-2 overflow-y-auto pr-1">
            {results.length === 0 ? (
              <div className="py-10 text-center">
                <p className="text-sm text-slate-600">
                  Nothing matches &ldquo;{query.trim()}&rdquo;
                </p>
                <p className="mt-1 text-xs text-slate-400">
                  Try a shorter phrase, or check the spelling of a name.
                </p>
              </div>
            ) : (
              results.map((r, i) => (
                <div
                  key={`${r.ts}-${i}`}
                  className="rounded-lg border border-slate-200 bg-white p-3 text-sm transition-colors hover:border-slate-300"
                >
                  <div className="mb-1.5 flex items-center gap-2 text-xs">
                    <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono tabular-nums text-slate-600">
                      {fmtTs(r.ts)}
                    </span>
                    <span className="font-medium text-slate-700">{r.speaker}</span>
                    {r.type && (
                      <span className="ml-auto rounded bg-indigo-50 px-1.5 py-0.5 font-medium text-indigo-700">
                        {r.type}
                      </span>
                    )}
                  </div>
                  <p className="leading-relaxed text-slate-700">
                    <Highlighted text={r.text} query={isPresetQuery ? '' : query} />
                  </p>
                </div>
              ))
            )}
          </div>
        </>
      )}

      {q.length > 0 && q.length < 2 && (
        <p className="mt-3 text-xs text-slate-400">Type at least 2 characters to search.</p>
      )}

      {q.length === 0 && (
        <p className="mt-4 text-xs text-slate-400">
          Search across {(index.segments ?? []).length.toLocaleString()} transcript segments.
        </p>
      )}
    </div>
  );
}
