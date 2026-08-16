'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ProfileSelect } from '@/components/ProfileSelect';
import { UploadZone } from '@/components/UploadZone';
import { checkSetup, uploadJob } from '@/lib/api';
import type { SetupCheckResponse } from '@/lib/types';

const TIMESTAMP_MODES = [
  { value: 'none', label: 'None' },
  { value: 'speaker', label: 'Per speaker' },
  { value: 'paragraph', label: 'Per paragraph' },
  { value: 'minute', label: 'Per minute' },
];

const WHISPER_MODES = [
  {
    value: 'fast',
    label: 'Fast',
    description: 'Distil-Whisper · no alignment',
    expected: '5–10 min',
  },
  {
    value: 'balanced',
    label: 'Balanced',
    description: 'Whisper turbo · word timestamps',
    expected: '20–30 min',
  },
  {
    value: 'high_accuracy',
    label: 'High Accuracy',
    description: 'Whisper large-v3 · beam search',
    expected: '~45 min',
  },
  {
    value: 'archive',
    label: 'Archive',
    description: 'large-v3 · AI grammar pass',
    expected: '~90 min',
  },
];

// Both modes run the same Parakeet TDT model — mode only changes which
// post-processors run (see PIPELINE_MODES in config/settings.py). Quick is
// the minimum-processing, fastest safe path; Standard adds an AI grammar
// correction pass on top for cleaner prose at the cost of extra time.
const PARAKEET_MODES = [
  {
    value: 'balanced',
    label: 'Quick',
    description: 'Speaker diarization · meeting intelligence',
    expected: '< 2 min',
  },
  {
    value: 'archive',
    label: 'Standard',
    description: 'Quick + AI grammar correction',
    expected: '3–5 min',
  },
];

export default function HomePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState('generic');
  const [timestamps, setTimestamps] = useState('none');
  const [mode, setMode] = useState('balanced');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [setup, setSetup] = useState<SetupCheckResponse | null>(null);
  const [setupError, setSetupError] = useState<string | null>(null);

  useEffect(() => {
    checkSetup()
      .then((s) => {
        setSetup(s);
        setSetupError(null);
      })
      .catch((err) => {
        // Previously swallowed. Silently falling back to defaults made a real
        // outage (unreachable API) look like a working page showing the wrong
        // backend's options — the page would offer Whisper modes while the
        // server ran Parakeet.
        setSetupError(err instanceof Error ? err.message : 'Cannot reach the API');
      });
  }, []);

  const isParakeet = setup?.asr_backend === 'parakeet';
  const activeModes = isParakeet ? PARAKEET_MODES : WHISPER_MODES;
  // Until /setup/check answers we don't know which backend is running, so the
  // mode list is a guess. Showing skeletons avoids flashing Whisper's four
  // modes and then swapping them for Parakeet's two a moment later.
  const backendUnknown = setup === null && setupError === null;

  const gpuLabel = (() => {
    const gpu = setup?.gpu;
    // Report what the backend actually detected. Previously this said "CUDA"
    // even when no CUDA device was present, implying GPU acceleration on a
    // machine running entirely on CPU.
    if (!gpu?.available) return 'CPU (no GPU detected)';
    const vram = gpu.vram_total_gb ? `${gpu.vram_total_gb.toFixed(0)} GB` : '';
    // Shorten common GPU names: "NVIDIA GeForce RTX 3060 Ti" → "RTX 3060 Ti"
    const name = (gpu.name ?? 'GPU').replace(/^NVIDIA\s+(GeForce\s+)?/i, '').trim();
    return vram ? `${name} · ${vram}` : name;
  })();

  // Use profiles from setup, or fallback
  const profiles = setup?.profiles?.length ? setup.profiles : ['generic', 'banking'];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setUploading(true);
    try {
      const { job_id } = await uploadJob(file, profile, timestamps, false, mode);
      router.push(`/jobs/${job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <span className="font-semibold text-slate-900">Transcript Engine</span>
          </div>
          <Link
            href="/setup"
            className="rounded-md px-2 py-1 text-sm text-slate-500 transition-colors hover:text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          >
            Setup
          </Link>
        </div>
      </header>

      {/* Main */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12">
        <div className="w-full max-w-xl">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Transcribe a meeting</h1>
            <p className="text-slate-500">Local AI — everything runs on your machine. No data leaves.</p>
          </div>

          {/* Setup warnings */}
          {setup && !setup.ffmpeg && (
            <div className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
              FFmpeg not found. Install it with <code className="bg-red-100 px-1 rounded">brew install ffmpeg</code>
            </div>
          )}
          {setup && !setup.hf_token.present && (
            <div className="mb-4 p-3 rounded-lg bg-amber-50 border border-amber-200 text-sm text-amber-700">
              HuggingFace token not set — speaker identification will be skipped.{' '}
              <Link href="/setup" className="font-medium underline underline-offset-2 hover:text-amber-900">Configure</Link>
            </div>
          )}

          {setupError && (
            <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-4">
              <p className="text-sm font-medium text-amber-800 mb-1">
                Cannot reach the transcription server
              </p>
              <p className="text-xs text-amber-700 leading-relaxed">
                Options below are defaults and may not match what the server actually runs.
                Check that the backend is running and reachable from this page. ({setupError})
              </p>
            </div>
          )}

          {setup?.gpu?.compatibility_error && (
            <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-4">
              <p className="text-sm font-medium text-red-800 mb-1">
                GPU not usable by the installed PyTorch build
              </p>
              <p className="text-xs text-red-700 leading-relaxed">
                {setup.gpu.compatibility_error}
              </p>
            </div>
          )}

          <Card className="p-6 bg-white shadow-sm border-slate-200">
            <form onSubmit={handleSubmit} className="space-y-6">
              <UploadZone onFile={setFile} maxUploadMb={setup?.max_upload_mb} />

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="block text-sm font-medium text-slate-700">Processing mode</label>
                  {isParakeet && (
                    <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-full px-2 py-0.5">
                      <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
                      </svg>
                      Parakeet TDT · {gpuLabel}
                    </span>
                  )}
                </div>
                {backendUnknown ? (
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" aria-busy="true">
                    {[0, 1].map((i) => (
                      <div
                        key={i}
                        className="h-[62px] animate-pulse rounded-lg border border-slate-200 bg-slate-50"
                      />
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" role="radiogroup" aria-label="Processing mode">
                    {activeModes.map((m) => {
                      const selected = mode === m.value;
                      return (
                        <button
                          key={m.value}
                          type="button"
                          role="radio"
                          aria-checked={selected}
                          onClick={() => setMode(m.value)}
                          className={`cursor-pointer rounded-lg border px-3 py-2.5 text-left transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 ${
                            selected
                              ? 'border-indigo-500 bg-indigo-50 ring-1 ring-indigo-500'
                              : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className={`text-sm font-medium ${selected ? 'text-indigo-700' : 'text-slate-800'}`}>
                              {m.label}
                            </span>
                            <span className="shrink-0 text-xs tabular-nums text-slate-400">{m.expected}</span>
                          </div>
                          <p className="mt-0.5 text-xs text-slate-500">{m.description}</p>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4">
                <ProfileSelect
                  profiles={profiles}
                  value={profile}
                  onChange={setProfile}
                />

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-2">Timestamps</label>
                  <select
                    className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-slate-800 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    value={timestamps}
                    onChange={(e) => setTimestamps(e.target.value)}
                  >
                    {TIMESTAMP_MODES.map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                  <p className="mt-1.5 text-xs text-slate-500">Add timestamps to the transcript</p>
                </div>
              </div>

              {error && (
                <p className="text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">{error}</p>
              )}

              <Button
                type="submit"
                disabled={!file || uploading}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2.5 rounded-lg transition-colors disabled:opacity-50"
              >
                {uploading ? (
                  <span className="flex items-center gap-2 justify-center">
                    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    Uploading…
                  </span>
                ) : 'Transcribe'}
              </Button>
            </form>
          </Card>

          <p className="text-center text-xs text-slate-400 mt-6">
            Supports AAC · MP3 · WAV · M4A · FLAC · MP4 · MOV · OGG · up to {setup?.max_upload_mb ?? 2000} MB
          </p>
        </div>
      </main>

      <footer className="py-4 text-center text-xs text-slate-400 border-t border-slate-100">
        Transcript Engine · <Link href="/setup" className="underline-offset-2 hover:text-slate-600 hover:underline">Setup</Link>
      </footer>
    </div>
  );
}
