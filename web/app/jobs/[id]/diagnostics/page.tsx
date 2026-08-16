'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { fetchDiagnostics } from '@/lib/api';
import type { DiagnosticsResponse } from '@/lib/types';

function fmtDur(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${String(sec).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

export default function DiagnosticsPage() {
  const params = useParams();
  const jobId = typeof params.id === 'string' ? params.id : null;
  const [data, setData] = useState<DiagnosticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    fetchDiagnostics(jobId)
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setFetchError(String(e.message)); setLoading(false); });
  }, [jobId]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-3 text-slate-500" aria-busy="true">
        <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        Loading diagnostics…
      </div>
    );
  }

  // Every terminal state below offers a way out. Previously a failed fetch
  // rendered bare red text with no navigation, stranding the user on a dead end.
  if (fetchError || !data) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-slate-100">
          <svg className="h-7 w-7 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
          </svg>
        </div>
        <h1 className="mb-1 text-lg font-semibold text-slate-800">Diagnostics unavailable</h1>
        <p className="mb-6 max-w-md text-sm text-slate-500">
          {fetchError ?? 'No diagnostics report was found for this job.'}
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          {jobId && (
            <Link
              href={`/jobs/${jobId}`}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
            >
              Back to results
            </Link>
          )}
          <Link
            href="/"
            className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            New transcription
          </Link>
        </div>
      </div>
    );
  }

  const report = data.processing_report;
  const resources = data.resource_usage;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-3">
        <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        </div>
        <span className="font-semibold text-slate-800 text-sm">Diagnostics</span>
        <span className="text-slate-300 mx-1">·</span>
        <span className="text-sm text-slate-500 font-mono">{jobId}</span>
        <Link
          href={`/jobs/${jobId}`}
          className="ml-auto rounded-md px-2 py-1 text-sm text-indigo-600 underline-offset-2 transition-colors hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          Back to results
        </Link>
      </header>

      <div className="max-w-4xl mx-auto p-6 space-y-6">
        {!report && (
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-700">
            No processing report available yet. The job may still be running or may have failed
            before writing reports.
          </div>
        )}

        {report && (
          <>
            {/* Summary */}
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <h2 className="font-semibold text-slate-800 mb-3">Summary</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div>
                  <p className="text-xs text-slate-500">Total Time</p>
                  <p className="text-lg font-semibold text-slate-800">{fmtDur(report.total_s)}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Audio Duration</p>
                  <p className="text-lg font-semibold text-slate-800">
                    {fmtDur(report.audio_duration_s)}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Processing RTF</p>
                  <p className="text-lg font-semibold text-slate-800">
                    {report.audio_duration_s > 0 && report.total_s > 0
                      ? `${(report.audio_duration_s / report.total_s).toFixed(1)}×`
                      : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Status</p>
                  <p
                    className={`text-lg font-semibold ${
                      report.status === 'completed' ? 'text-green-600' : 'text-red-600'
                    }`}
                  >
                    {report.status.toUpperCase()}
                  </p>
                </div>
              </div>
              <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-xs text-slate-500">Started</p>
                  <p className="text-slate-700">{report.started_at}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Completed</p>
                  <p className="text-slate-700">{report.completed_at}</p>
                </div>
                <div>
                  <p className="text-xs text-slate-500">Upload</p>
                  <p className="text-slate-700">{fmtDur(report.upload_s)}</p>
                </div>
              </div>
            </div>

            {/* Stage timings */}
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <h2 className="font-semibold text-slate-800 mb-3">Stage Timings</h2>
              <div className="space-y-3">
                {Object.entries(report.stages)
                  .sort(([, a], [, b]) => (a.start_s ?? 0) - (b.start_s ?? 0))
                  .map(([stage, stageData]) => (
                    <div key={stage}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-700">{stage}</span>
                        <span className="text-slate-500 tabular-nums">
                          {fmtDur(stageData.duration_s)}{' '}
                          <span className="text-slate-400">({stageData.pct.toFixed(1)}%)</span>
                        </span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-indigo-500 rounded-full transition-all"
                          style={{ width: `${Math.min(stageData.pct, 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
              </div>
            </div>

            {/* Models */}
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <h2 className="font-semibold text-slate-800 mb-3">Models Used</h2>
              <dl className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <dt className="text-xs text-slate-500 mb-0.5">ASR Model</dt>
                  <dd className="font-mono text-slate-800 text-xs bg-slate-50 px-2 py-1 rounded">
                    {report.models.asr}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-slate-500 mb-0.5">Device</dt>
                  <dd className="font-mono text-slate-800 text-xs bg-slate-50 px-2 py-1 rounded">
                    {report.models.device}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-slate-500 mb-0.5">Diarization</dt>
                  <dd className="font-mono text-slate-800 text-xs bg-slate-50 px-2 py-1 rounded">
                    {report.models.diarization}
                  </dd>
                </div>
              </dl>
            </div>

            {/* Artifacts */}
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <h2 className="font-semibold text-slate-800 mb-1">
                Artifacts{' '}
                <span className="text-sm font-normal text-slate-500">
                  ({report.output_validation.present}/{report.output_validation.total})
                </span>
              </h2>
              {report.output_validation.missing.length > 0 && (
                <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                  Missing: {report.output_validation.missing.join(', ')}
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-1.5">
                {Object.entries(report.output_validation.artifacts ?? {}).map(([name, info]) => {
                  const ok = (info as { status: string }).status === 'ok';
                  const bytes = (info as { size_bytes?: number }).size_bytes;
                  return (
                    <div
                      key={name}
                      className={`text-xs px-2 py-1.5 rounded flex items-center gap-1.5 ${
                        ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                      }`}
                    >
                      <span>{ok ? '✓' : '✗'}</span>
                      <span className="truncate" title={name}>
                        {name}
                        {ok && bytes != null && (
                          <span className="text-green-500 ml-1">
                            ({(bytes / 1024).toFixed(0)}k)
                          </span>
                        )}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Temp cleanup */}
            <div className="bg-white rounded-lg border border-slate-200 p-4">
              <h2 className="font-semibold text-slate-800 mb-3">Temp File Cleanup</h2>
              <div className="flex gap-6 text-sm">
                <span
                  className={
                    report.temp_validation.audio_deleted ? 'text-green-600' : 'text-red-600'
                  }
                >
                  {report.temp_validation.audio_deleted ? '✓' : '✗'} Audio file deleted
                </span>
                <span
                  className={
                    report.temp_validation.temp_dir_deleted
                      ? 'text-green-600'
                      : 'text-amber-600'
                  }
                >
                  {report.temp_validation.temp_dir_deleted ? '✓' : '~'} Temp directory cleared
                </span>
              </div>
            </div>
          </>
        )}

        {/* Resource usage */}
        {resources?.summary && (
          <div className="bg-white rounded-lg border border-slate-200 p-4">
            <h2 className="font-semibold text-slate-800 mb-3">Resource Usage</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mb-4">
              <div>
                <p className="text-xs text-slate-500">CPU Peak</p>
                <p className="font-semibold text-slate-800">{resources.summary.cpu_peak}%</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">CPU Avg</p>
                <p className="font-semibold text-slate-800">{resources.summary.cpu_avg}%</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">RAM Peak</p>
                <p className="font-semibold text-slate-800">
                  {resources.summary.ram_peak_gb.toFixed(1)} GB
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">RAM Avg</p>
                <p className="font-semibold text-slate-800">
                  {resources.summary.ram_avg_gb.toFixed(1)} GB
                </p>
              </div>
            </div>
            {resources.samples.length > 0 && (
              <div>
                <p className="text-xs text-slate-500 mb-2">
                  CPU over time ({resources.samples.length} samples, every {resources.interval_s}s)
                </p>
                <div className="flex items-end gap-0.5 h-12">
                  {resources.samples.map((s, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-indigo-400 rounded-t min-w-0"
                      style={{
                        height: `${Math.min((s.cpu_pct / Math.max(resources.summary.cpu_peak, 1)) * 100, 100)}%`,
                        opacity: 0.6 + 0.4 * (s.cpu_pct / Math.max(resources.summary.cpu_peak, 1)),
                      }}
                      title={`${s.stage}: CPU ${s.cpu_pct}%, RAM ${s.ram_gb.toFixed(1)}GB`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
