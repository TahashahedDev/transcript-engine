'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { fetchArtifactText, fetchSearchIndex, getArtifactUrl } from '@/lib/api';
import { copyText } from '@/lib/clipboard';
import type { Job, SearchIndex } from '@/lib/types';
import { MarkdownViewer } from './MarkdownViewer';
import { SearchBox } from './SearchBox';

function fmtSeconds(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}m ${sec}s`;
}

interface ArtifactTabProps {
  jobId: string;
  filename: string;
  isJson?: boolean;
}

function LoadingLines() {
  return (
    <div className="space-y-3 p-6" aria-busy="true" aria-label="Loading content">
      {[100, 92, 96, 70, 88, 94, 60].map((w, i) => (
        <div
          key={i}
          className="h-3.5 animate-pulse rounded bg-slate-100"
          style={{ width: `${w}%`, animationDelay: `${i * 60}ms` }}
        />
      ))}
    </div>
  );
}

/**
 * Outer shell owns only the retry counter. Bumping it changes the child's
 * `key`, so React remounts the child with fresh state — that resets to the
 * loading view without a setState-in-effect.
 */
function ArtifactTab(props: ArtifactTabProps) {
  const [attempt, setAttempt] = useState(0);
  return (
    <ArtifactContent
      key={`${props.jobId}|${props.filename}|${attempt}`}
      {...props}
      onRetry={() => setAttempt((a) => a + 1)}
    />
  );
}

function ArtifactContent({
  jobId,
  filename,
  isJson,
  onRetry,
}: ArtifactTabProps & { onRetry: () => void }) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchArtifactText(jobId, filename)
      .then((t) => {
        if (cancelled) return;
        setContent(t);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        // Distinguish "still loading" from "this failed" — the old code showed
        // a flat "File not available." with no way to recover.
        setFailed(true);
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, [jobId, filename]);

  if (loading) return <LoadingLines />;

  if (failed) {
    return (
      <div className="p-10 text-center">
        <p className="text-sm text-slate-600">Could not load this file.</p>
        <p className="mt-1 text-xs text-slate-400">
          It may not have been generated for this job, or the server is unreachable.
        </p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 cursor-pointer rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          Try again
        </button>
      </div>
    );
  }

  if (!content || !content.trim()) {
    return (
      <div className="p-10 text-center">
        <p className="text-sm text-slate-600">Nothing here for this recording.</p>
        <p className="mt-1 text-xs text-slate-400">
          This section is generated only when the meeting contains relevant content.
        </p>
      </div>
    );
  }

  if (isJson) {
    return (
      <pre className="overflow-auto whitespace-pre-wrap p-4 font-mono text-xs text-slate-700">
        {content}
      </pre>
    );
  }

  return (
    <div className="p-6">
      <MarkdownViewer content={content} />
    </div>
  );
}

/**
 * Individual file downloads, grouped by what the user is trying to do.
 *
 * Every one of these is written for each completed job, so the menu is a
 * static list rather than a fetch of the artifact index — one less request and
 * one less failure mode on a page whose job is already done.
 */
const DOWNLOAD_GROUPS: Array<{
  label: string;
  items: Array<{ file: string; label: string; hint: string }>;
}> = [
  {
    label: 'Transcript',
    items: [
      { file: 'transcript.md', label: 'Markdown', hint: 'Formatted, with speaker headings' },
      { file: 'transcript.txt', label: 'Plain text', hint: 'Paste anywhere' },
      { file: 'transcript.json', label: 'JSON', hint: 'Word timings and metadata' },
    ],
  },
  {
    label: 'Subtitles',
    items: [
      { file: 'transcript.srt', label: 'SubRip', hint: 'Video editors, .srt' },
      { file: 'transcript.vtt', label: 'WebVTT', hint: 'Web players, .vtt' },
    ],
  },
  {
    label: 'Meeting notes',
    items: [
      { file: 'transcript.summary.md', label: 'Summary', hint: 'Key points' },
      { file: 'transcript.action_items.md', label: 'Action items', hint: 'Who owes what' },
      { file: 'transcript.decisions.md', label: 'Decisions', hint: 'What was agreed' },
    ],
  },
];

function DownloadMenu({ jobId }: { jobId: string }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close on outside click and on Escape — a menu you can only dismiss by
  // picking something is a trap, especially on touch.
  useEffect(() => {
    if (!open) return;
    function onPointerDown(e: MouseEvent | TouchEvent) {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setOpen(false);
    }
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('touchstart', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('touchstart', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);

  return (
    <div className="relative" ref={containerRef}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
      >
        Download file
        <svg
          className={`h-3.5 w-3.5 transition-transform duration-150 ${open ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute left-0 top-full z-20 mt-1 max-h-[70vh] w-72 overflow-y-auto rounded-lg border border-slate-200 bg-white py-1 shadow-lg"
        >
          {DOWNLOAD_GROUPS.map((group) => (
            <div key={group.label}>
              <p className="px-3 pb-1 pt-2 text-[0.7rem] font-semibold uppercase tracking-wide text-slate-400">
                {group.label}
              </p>
              {group.items.map((item) => (
                <a
                  key={item.file}
                  role="menuitem"
                  href={getArtifactUrl(jobId, item.file)}
                  download
                  onClick={() => setOpen(false)}
                  className="flex cursor-pointer items-baseline justify-between gap-3 px-3 py-2 transition-colors hover:bg-slate-50 focus-visible:bg-slate-50 focus-visible:outline-none"
                >
                  <span className="text-sm text-slate-700">{item.label}</span>
                  <span className="shrink-0 text-xs text-slate-400">{item.hint}</span>
                </a>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const PRIMARY_TABS = [
  { value: 'transcript', label: 'Transcript' },
  { value: 'summary', label: 'Summary' },
  { value: 'actions', label: 'Action Items' },
  { value: 'decisions', label: 'Decisions' },
  { value: 'questions', label: 'Questions' },
  { value: 'timeline', label: 'Timeline' },
  { value: 'search', label: 'Search' },
];

const ADVANCED_TABS = [
  { value: 'review', label: 'Review' },
  { value: 'quality', label: 'Quality' },
  { value: 'entities', label: 'Entities' },
  { value: 'metrics', label: 'Metrics' },
];

const ADVANCED_VALUES = new Set(ADVANCED_TABS.map((t) => t.value));

interface Props {
  job: Job;
}

type CopyState = 'idle' | 'copying' | 'copied' | 'failed';

export function ResultsView({ job }: Props) {
  const [searchIndex, setSearchIndex] = useState<SearchIndex | null>(null);
  const [searchIndexFailed, setSearchIndexFailed] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [activeTab, setActiveTab] = useState('transcript');
  const [copyState, setCopyState] = useState<CopyState>('idle');

  const m = job.metrics;
  const jobId = job.job_id;

  useEffect(() => {
    let cancelled = false;
    fetchSearchIndex(jobId)
      .then((idx) => { if (!cancelled) setSearchIndex(idx); })
      // Previously swallowed, which left the Search tab stuck on
      // "Loading search index…" forever with no explanation.
      .catch(() => { if (!cancelled) setSearchIndexFailed(true); });
    return () => { cancelled = true; };
  }, [jobId]);

  function handleTabChange(tab: string) {
    setActiveTab(tab);
    if (ADVANCED_VALUES.has(tab)) setShowAdvanced(true);
  }

  async function copyTranscript() {
    setCopyState('copying');
    try {
      const text = await fetchArtifactText(jobId, 'transcript.md');
      const ok = await copyText(text);
      setCopyState(ok ? 'copied' : 'failed');
    } catch {
      setCopyState('failed');
    }
    // Both outcomes reset, so the button never gets stuck in a terminal state.
    setTimeout(() => setCopyState('idle'), 2500);
  }

  const bundleUrl = getArtifactUrl(jobId, 'transcript_bundle.zip');

  // shrink-0 + whitespace-nowrap keep each tab intact inside the scrolling row
  // instead of squeezing or breaking labels like "Action Items".
  const triggerClass =
    'shrink-0 cursor-pointer whitespace-nowrap rounded-md px-3 py-1.5 text-xs transition-colors ' +
    'hover:bg-slate-50 hover:text-slate-800 ' +
    'data-[state=active]:bg-indigo-50 data-[state=active]:font-medium data-[state=active]:text-indigo-700';

  return (
    <div className="flex flex-col h-full">
      {/* Stats bar */}
      {m && (
        <div className="flex flex-wrap gap-3 px-6 py-4 border-b border-slate-100 bg-slate-50">
          {m.audio_duration_seconds != null && (
            <Badge variant="secondary">Duration: {fmtSeconds(m.audio_duration_seconds)}</Badge>
          )}
          {m.processing_time_seconds != null && (
            <Badge variant="secondary">Processed in {fmtSeconds(m.processing_time_seconds)}</Badge>
          )}
          {m.word_count != null && (
            <Badge variant="secondary">{m.word_count.toLocaleString()} words</Badge>
          )}
          {m.speaker_count != null && (
            <Badge variant="secondary">{m.speaker_count} speaker{m.speaker_count !== 1 ? 's' : ''}</Badge>
          )}
          {m.action_items != null && m.action_items > 0 && (
            <Badge className="bg-indigo-100 text-indigo-700 hover:bg-indigo-100">{m.action_items} action item{m.action_items !== 1 ? 's' : ''}</Badge>
          )}
          {m.decisions != null && m.decisions > 0 && (
            <Badge className="bg-emerald-100 text-emerald-700 hover:bg-emerald-100">{m.decisions} decision{m.decisions !== 1 ? 's' : ''}</Badge>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-100 px-6 py-3">
        <a
          href={bundleUrl}
          download
          className="inline-flex cursor-pointer items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        >
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          Download all
          <span className="text-xs font-normal text-indigo-200">.zip</span>
        </a>

        <DownloadMenu jobId={jobId} />
        <button
          type="button"
          onClick={copyTranscript}
          disabled={copyState === 'copying'}
          aria-live="polite"
          className={`inline-flex cursor-pointer items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-60 ${
            copyState === 'failed'
              ? 'border-red-300 bg-red-50 text-red-700'
              : 'border-slate-300 bg-white text-slate-700 hover:bg-slate-50'
          }`}
        >
          {copyState === 'copied' && (
            <>
              <svg className="h-4 w-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
              Copied
            </>
          )}
          {copyState === 'failed' && (
            <>
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
              </svg>
              Copy failed — use Download
            </>
          )}
          {copyState === 'copying' && (
            <>
              <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Copying…
            </>
          )}
          {copyState === 'idle' && (
            <>
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Copy transcript
            </>
          )}
        </button>
        <Link
          href="/"
          className="ml-auto inline-flex items-center gap-2 px-4 py-2 text-slate-500 text-sm hover:text-slate-700 transition-colors"
        >
          ← New transcription
        </Link>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={handleTabChange} className="flex-1 flex flex-col">
        <div className="border-b border-slate-100 px-6 pt-2">
          <div className="flex items-center gap-2">
            {/*
              Scrolls horizontally rather than wrapping. Wrapping pushed the
              last one or two tabs onto a second row where the active tab
              looked detached from the group; a single scrolling row keeps the
              set readable at any width, including mobile.
            */}
            <TabsList className="h-auto min-w-0 flex-1 flex-nowrap justify-start gap-1 overflow-x-auto rounded-none bg-transparent p-0 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              {PRIMARY_TABS.map(({ value, label }) => (
                <TabsTrigger key={value} value={value} className={triggerClass}>
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
            {/* Pinned right and kept out of the wrapping row so it never lands
                mid-row next to a stray tab. */}
            <button
              type="button"
              onClick={() => setShowAdvanced((s) => !s)}
              aria-expanded={showAdvanced}
              className="mt-0.5 inline-flex shrink-0 cursor-pointer items-center gap-1 rounded-md border border-transparent px-2.5 py-1.5 text-xs text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-50 hover:text-slate-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              Advanced
              <svg
                className={`h-3 w-3 transition-transform duration-150 ${showAdvanced ? 'rotate-180' : ''}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
          {showAdvanced && (
            <TabsList className="mb-2 mt-1 h-auto flex-wrap justify-start gap-1 rounded-md border border-slate-200 bg-slate-50 p-1">
              {ADVANCED_TABS.map(({ value, label }) => (
                <TabsTrigger key={value} value={value} className={triggerClass}>
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>
          )}
        </div>

        <ScrollArea className="flex-1">
          <TabsContent value="transcript" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.md" />
          </TabsContent>
          <TabsContent value="summary" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.summary.md" />
          </TabsContent>
          <TabsContent value="actions" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.action_items.md" />
          </TabsContent>
          <TabsContent value="decisions" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.decisions.md" />
          </TabsContent>
          <TabsContent value="questions" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.questions.md" />
          </TabsContent>
          <TabsContent value="timeline" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.timeline.md" />
          </TabsContent>
          <TabsContent value="search" className="mt-0">
            <div className="p-6">
              {searchIndex ? (
                <SearchBox index={searchIndex} />
              ) : searchIndexFailed ? (
                <div className="py-10 text-center">
                  <p className="text-sm text-slate-600">Search is unavailable for this job.</p>
                  <p className="mt-1 text-xs text-slate-400">
                    The search index could not be loaded. You can still read and download the
                    transcript from the other tabs.
                  </p>
                </div>
              ) : (
                <div className="space-y-3" aria-busy="true">
                  <div className="h-10 animate-pulse rounded-lg bg-slate-100" />
                  <div className="h-3 w-32 animate-pulse rounded bg-slate-100" />
                </div>
              )}
            </div>
          </TabsContent>
          <TabsContent value="review" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.review.md" />
          </TabsContent>
          <TabsContent value="quality" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.quality.md" />
          </TabsContent>
          <TabsContent value="entities" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.entities.json" isJson />
          </TabsContent>
          <TabsContent value="metrics" className="mt-0">
            <ArtifactTab jobId={jobId} filename="transcript.metrics.json" isJson />
          </TabsContent>
        </ScrollArea>
      </Tabs>
    </div>
  );
}
