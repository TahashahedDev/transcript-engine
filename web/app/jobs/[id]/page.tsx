'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { ProgressView } from '@/components/ProgressView';
import { ResultsView } from '@/components/ResultsView';
import { useJob } from '@/hooks/useJob';
import { useProgress } from '@/hooks/useProgress';
import { checkSetup } from '@/lib/api';

export default function JobPage() {
  const params = useParams();
  const jobId = typeof params.id === 'string' ? params.id : null;

  const { currentStage, fraction, completed, error, connectionLost, audioDuration, cpuPct, ramGb, stageStartedAt, pipelineDecisions, partialTranscript } = useProgress(jobId);
  const { job, loading } = useJob(jobId, completed);

  // Which ASR backend is actually running — drives model-name copy in
  // ProgressView so it never claims Whisper is loading during a Parakeet job.
  const [asrBackend, setAsrBackend] = useState<string | null>(null);
  useEffect(() => {
    checkSetup().then((s) => setAsrBackend(s.asr_backend)).catch(() => {});
  }, []);

  // Track if we should show results — either job completed, or SSE says completed
  const showResults =
    job?.status === 'completed' ||
    (completed && job != null);

  const showError = job?.status === 'failed' || (error != null && !showResults);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-3 text-slate-500">
          <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading…
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-slate-100">
          <svg className="h-7 w-7 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </div>
        <h1 className="mb-1 text-lg font-semibold text-slate-800">This job no longer exists</h1>
        <p className="mb-6 max-w-sm text-sm text-slate-500">
          Jobs are kept in memory, so the link stops working if the server restarted or the
          job was cleaned up.
        </p>
        <Link
          href="/"
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
        >
          Start a new transcription
        </Link>
      </div>
    );
  }

  if (showResults) {
    return (
      <div className="min-h-screen flex flex-col">
        <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <span className="font-semibold text-slate-800 text-sm">Transcript Engine</span>
          <span className="text-slate-300 mx-1">·</span>
          <span className="text-sm text-slate-500 truncate">{job.audio_filename}</span>
          <span className="ml-auto flex items-center gap-3">
            <Link
              href={`/jobs/${job.job_id}/diagnostics`}
              className="rounded-md px-2 py-1 text-xs text-slate-500 underline-offset-2 transition-colors hover:text-indigo-600 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
            >
              Diagnostics
            </Link>
            <span className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded-full border border-emerald-200">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
              Complete
            </span>
          </span>
        </header>
        <div className="flex-1 flex flex-col">
          <ResultsView job={job} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </div>
        <span className="font-semibold text-slate-800 text-sm">Transcript Engine</span>
      </header>
      <div className="flex-1 px-4">
        <ProgressView
          currentStage={showError ? 'Error' : currentStage}
          fraction={fraction}
          error={showError ? (job.error ?? error ?? 'Transcription failed') : null}
          audioFilename={job.audio_filename}
          audioDuration={audioDuration ?? undefined}
          cpuPct={cpuPct}
          ramGb={ramGb}
          stageStartedAt={stageStartedAt}
          pipelineDecisions={pipelineDecisions}
          partialTranscript={partialTranscript}
          mode={job.mode}
          asrBackend={asrBackend ?? undefined}
          connectionLost={connectionLost}
          jobId={job.job_id}
        />
      </div>
    </div>
  );
}
