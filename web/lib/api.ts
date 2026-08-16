import { resolveApiBase } from './apiBase';
import type { ArtifactsResponse, DiagnosticsResponse, Job, SearchIndex, SetupCheckResponse } from './types';

// Resolved per call, never cached at module scope: a 'use client' module is
// also evaluated on the server during SSR, where window is undefined and the
// localhost fallback applies. Capturing that value in a module-level const
// leaks the server's answer into the browser bundle, which then calls
// localhost instead of the host the page was served from.
const apiBase = () => resolveApiBase();

export async function uploadJob(
  file: File,
  profile: string,
  timestamps: string,
  highlightConfidence: boolean,
  mode: string = 'balanced',
): Promise<{ job_id: string }> {
  const form = new FormData();
  form.append('file', file);
  form.append('profile', profile);
  form.append('timestamps', timestamps);
  form.append('highlight_confidence', String(highlightConfidence));
  form.append('mode', mode);

  const res = await fetch(`${apiBase()}/api/jobs`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Upload failed');
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<Job> {
  const res = await fetch(`${apiBase()}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error('Job not found');
  return res.json();
}

export async function getArtifacts(jobId: string): Promise<ArtifactsResponse> {
  const res = await fetch(`${apiBase()}/api/jobs/${jobId}/artifacts`);
  if (!res.ok) throw new Error('Failed to list artifacts');
  return res.json();
}

export async function fetchArtifactText(jobId: string, filename: string): Promise<string> {
  const res = await fetch(`${apiBase()}/api/jobs/${jobId}/artifacts/${filename}`);
  if (!res.ok) throw new Error(`Failed to fetch ${filename}`);
  return res.text();
}

export async function fetchSearchIndex(jobId: string): Promise<SearchIndex> {
  const text = await fetchArtifactText(jobId, 'transcript.index.json');
  return JSON.parse(text) as SearchIndex;
}

export function getArtifactUrl(jobId: string, filename: string): string {
  return `${apiBase()}/api/jobs/${jobId}/artifacts/${filename}`;
}

export async function checkSetup(): Promise<SetupCheckResponse> {
  const res = await fetch(`${apiBase()}/api/setup/check`);
  if (!res.ok) throw new Error('Failed to check setup');
  return res.json();
}

export async function saveToken(token: string): Promise<{ saved: boolean; valid: boolean | null; error: string | null }> {
  const res = await fetch(`${apiBase()}/api/setup/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    return { saved: false, valid: null, error: err.detail ?? 'Failed to save token' };
  }
  return res.json();
}

export async function triggerModelDownload(): Promise<void> {
  await fetch(`${apiBase()}/api/setup/download-models`, { method: 'POST' });
}

export async function deleteJob(jobId: string): Promise<void> {
  await fetch(`${apiBase()}/api/jobs/${jobId}`, { method: 'DELETE' });
}

export async function fetchDiagnostics(jobId: string): Promise<DiagnosticsResponse> {
  const res = await fetch(`${apiBase()}/api/jobs/${jobId}/diagnostics`);
  if (!res.ok) throw new Error('Failed to fetch diagnostics');
  return res.json();
}
