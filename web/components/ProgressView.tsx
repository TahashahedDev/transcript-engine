'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { Progress } from '@/components/ui/progress';
import { STAGES, STAGE_EXPECTED_SECONDS } from '@/hooks/useProgress';

function LiveTranscriptPanel({
  segments,
}: {
  segments: Array<{ speaker: string; text: string; start: number }>;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [segments]);

  const wordCount = segments.reduce((n, s) => n + s.text.split(' ').length, 0);

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-600 flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse inline-block" />
          Live Transcript
        </span>
        <span className="text-xs text-slate-400 tabular-nums">
          {segments.length} segments · {wordCount.toLocaleString()} words
        </span>
      </div>
      <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3 space-y-2">
        {segments.map((seg) => (
          <div key={`${seg.speaker}-${seg.start}`} className="text-xs">
            <span className="font-medium text-indigo-600">{seg.speaker}</span>
            <span className="text-slate-400 ml-1">
              [{Math.floor(seg.start / 60)}:{String(Math.floor(seg.start % 60)).padStart(2, '0')}]
            </span>
            <p className="text-slate-700 mt-0.5 leading-relaxed">{seg.text}</p>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

// Whisper model names vary by mode; Parakeet uses one fixed model regardless
// of mode (see transcript_engine/transcription/parakeet_engine.py — the mode
// only changes which post-processors run, not the ASR model).
const WHISPER_MODEL_LABELS: Record<string, string> = {
  fast: 'Distil-Whisper',
  balanced: 'Whisper Large V3 Turbo',
  high_accuracy: 'Whisper Large V3',
  archive: 'Whisper Large V3',
};
const PARAKEET_MODEL_LABEL = 'Parakeet TDT';

function getStageDescriptions(mode: string, asrBackend?: string): Record<string, string> {
  const isParakeet = asrBackend === 'parakeet';
  const model = isParakeet ? PARAKEET_MODEL_LABEL : (WHISPER_MODEL_LABELS[mode] ?? 'Whisper');
  return {
    'Uploading': 'Sending your file to the server…',
    'Preprocessing Audio': 'Converting audio to 16 kHz mono WAV…',
    'Loading Model': `Loading ${model} into memory…`,
    'Transcribing': `Converting speech to text using ${model} — this is the longest step…`,
    'Aligning Words': isParakeet
      ? 'Not used — Parakeet TDT produces word timestamps natively.'
      : 'Aligning word-level timestamps with wav2vec2…',
    'Loading Diarization': 'Loading pyannote speaker identification model…',
    'Speaker Diarization': 'Identifying who said what…',
    'Processing': 'Cleaning up the transcript and applying corrections…',
    'Vocabulary Corrections': 'Applying domain-specific vocabulary corrections…',
    'Meeting Intelligence': 'Extracting action items, decisions, and key topics…',
    'Generating Files': 'Writing transcript and intelligence files…',
    'Creating Bundle': 'Packaging all files into a download bundle…',
    'Completed': 'Done!',
  };
}

function fmtSeconds(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  if (m === 0) return `${sec}s`;
  return `${m}m ${sec}s`;
}

interface Props {
  currentStage: string;
  fraction: number;
  error: string | null;
  audioFilename: string;
  audioDuration?: number;
  cpuPct?: number | null;
  ramGb?: number | null;
  stageStartedAt?: number;
  pipelineDecisions?: string[];
  partialTranscript?: Array<{ speaker: string; text: string; start: number }> | null;
  mode?: string;
  asrBackend?: string;
  connectionLost?: boolean;
  jobId?: string;
}

export function ProgressView({
  currentStage,
  fraction,
  error,
  audioFilename,
  audioDuration,
  cpuPct,
  ramGb,
  stageStartedAt,
  pipelineDecisions = [],
  partialTranscript = null,
  mode = 'balanced',
  asrBackend,
  connectionLost = false,
  jobId,
}: Props) {
  const isParakeet = asrBackend === 'parakeet';
  // Parakeet's TDT decoder produces word timestamps natively — it never runs
  // the wav2vec2 alignment step, regardless of mode. Whisper's "fast" mode is
  // the only Whisper case that skips it.
  const skipsAlignment = isParakeet || mode === 'fast';
  const skippedStages = skipsAlignment ? new Set(['Aligning Words']) : new Set<string>();
  const visibleStages = STAGES.filter((s) => !skippedStages.has(s));
  const currentIndex = visibleStages.indexOf(currentStage);
  const STAGE_DESCRIPTIONS = getStageDescriptions(mode, asrBackend);
  const pct = Math.round(fraction * 100);

  const [elapsed, setElapsed] = useState(0);
  const [stageElapsed, setStageElapsed] = useState(0);
  const startRef = useRef<number>(0);

  useEffect(() => { startRef.current = Date.now(); }, []);

  useEffect(() => {
    if (error) return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startRef.current) / 1000));
      setStageElapsed(stageStartedAt
        ? Math.floor((Date.now() - stageStartedAt) / 1000)
        : 0);
    }, 1000);
    return () => clearInterval(id);
  }, [error, stageStartedAt]);

  const expectedStageDuration = STAGE_EXPECTED_SECONDS[currentStage] ?? 60;
  const stageSlow = stageElapsed > expectedStageDuration && currentStage !== 'Completed';

  // Streaming pipeline + batch_size=1: ~1.5x RTF on M1 (midpoint between 1.93x cold and 1.12x sustained under thermal load)
  const estimatedTotal = audioDuration ? Math.ceil(audioDuration / 1.5) : null;
  const estimatedRemaining =
    estimatedTotal && fraction > 0.05
      ? Math.max(0, Math.ceil(estimatedTotal * (1 - fraction)))
      : null;

  if (error) {
    return (
      <div className="mx-auto mt-16 max-w-lg text-center" role="alert">
        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-red-50">
          <svg className="h-8 w-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
        </div>
        <h2 className="mb-2 text-xl font-semibold text-slate-800">Transcription failed</h2>
        <p className="mb-6 whitespace-pre-line text-sm leading-relaxed text-slate-600">{error}</p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/"
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
          >
            Try again
          </Link>
          {jobId && (
            // A failed job is exactly when the stage timings and error report
            // matter most, so surface them here instead of only on success.
            <Link
              href={`/jobs/${jobId}/diagnostics`}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              View diagnostics
            </Link>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-lg mx-auto mt-12">
      <div className="mb-8 text-center">
        <h2 className="text-xl font-semibold text-slate-800 mb-1">Processing</h2>
        <p className="text-sm text-slate-500 truncate">{audioFilename}</p>
        {audioDuration != null && (
          <p className="text-xs text-slate-400 mt-1">
            Audio: {fmtSeconds(Math.round(audioDuration))}
            {estimatedTotal && (
              <> · Estimated: ~{fmtSeconds(estimatedTotal)}</>
            )}
          </p>
        )}
      </div>

      <div className="mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span className="font-medium text-indigo-600">{currentStage}</span>
          <span className="text-slate-500 tabular-nums">
            {pct}%
            {elapsed > 0 && <> · {fmtSeconds(elapsed)} elapsed</>}
            {estimatedRemaining != null && estimatedRemaining > 5 && (
              <> · ~{fmtSeconds(estimatedRemaining)} left</>
            )}
          </span>
        </div>
        <Progress value={pct} className="h-2" />
        {STAGE_DESCRIPTIONS[currentStage] && (
          <p className="text-xs text-slate-400 mt-1.5">{STAGE_DESCRIPTIONS[currentStage]}</p>
        )}

        {connectionLost ? (
          <p className="text-xs text-amber-600 mt-1.5 flex items-start gap-1">
            <svg className="w-3.5 h-3.5 shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            </svg>
            <span>
              Lost the live progress connection. The job is still running on the server —
              this page will update automatically when it finishes.
            </span>
          </p>
        ) : stageSlow && (
          <p className="text-xs text-amber-600 mt-1.5 flex items-center gap-1">
            <svg className="w-3.5 h-3.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            </svg>
            This step is taking longer than expected — still working…
          </p>
        )}

        {(cpuPct != null || ramGb != null) && (
          <p className="text-xs text-slate-400 mt-1 tabular-nums">
            {cpuPct != null && <>CPU {cpuPct}%</>}
            {cpuPct != null && ramGb != null && <> · </>}
            {ramGb != null && <>RAM {ramGb.toFixed(1)} GB</>}
          </p>
        )}
      </div>

      <ul className="space-y-2">
        {visibleStages.map((stage, i) => {
          const isDone = i < currentIndex;
          const isCurrent = i === currentIndex;
          const isFuture = i > currentIndex;

          return (
            <li key={stage} className={`flex items-center gap-3 text-sm ${isFuture ? 'text-slate-400' : 'text-slate-700'}`}>
              <span className="w-5 shrink-0 flex items-center justify-center">
                {isDone ? (
                  <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                  </svg>
                ) : isCurrent ? (
                  <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                ) : (
                  <span className="w-2 h-2 rounded-full bg-slate-200" />
                )}
              </span>
              <span className={isCurrent ? 'font-medium text-indigo-700' : ''}>{stage}</span>
            </li>
          );
        })}
      </ul>

      {pipelineDecisions.length > 0 && (
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-xs font-medium text-blue-700 mb-1.5">Pipeline Decisions</p>
          <ul className="space-y-1">
            {pipelineDecisions.map((d) => (
              <li key={d} className="text-xs text-blue-600 flex items-start gap-1.5">
                <span className="shrink-0 mt-0.5">·</span>
                <span>{d}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {partialTranscript && partialTranscript.length > 0 && (
        <LiveTranscriptPanel segments={partialTranscript} />
      )}
    </div>
  );
}
