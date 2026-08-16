'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { Card } from '@/components/ui/card';
import { checkSetup, saveToken, triggerModelDownload } from '@/lib/api';
import type { ModelInfo, SetupCheckResponse } from '@/lib/types';

// ── Icons ─────────────────────────────────────────────────────────────────

function IconCheck() {
  return (
    <svg className="w-3.5 h-3.5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
    </svg>
  );
}

function IconWarn() {
  return (
    <svg className="w-3.5 h-3.5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01M12 3a9 9 0 100 18A9 9 0 0012 3z" />
    </svg>
  );
}

function IconError() {
  return (
    <svg className="w-3.5 h-3.5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function IconSpinner() {
  return (
    <svg className="animate-spin w-3.5 h-3.5 text-indigo-500" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

// ── Status row ────────────────────────────────────────────────────────────

type RowStatus = 'ok' | 'warn' | 'error' | 'loading';

function StatusRow({
  status,
  label,
  detail,
  children,
}: {
  status: RowStatus;
  label: string;
  detail?: string;
  children?: React.ReactNode;
}) {
  const icon =
    status === 'ok' ? <IconCheck /> :
    status === 'warn' ? <IconWarn /> :
    status === 'error' ? <IconError /> :
    <IconSpinner />;

  const bg =
    status === 'ok' ? 'bg-green-50' :
    status === 'warn' ? 'bg-amber-50' :
    status === 'error' ? 'bg-red-50' :
    'bg-slate-50';

  const labelColor =
    status === 'ok' ? 'text-slate-800' :
    status === 'warn' ? 'text-amber-900' :
    status === 'error' ? 'text-red-900' :
    'text-slate-600';

  return (
    <div className={`flex items-start gap-3 p-3 rounded-lg mb-2 last:mb-0 ${bg}`}>
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-medium ${labelColor}`}>{label}</p>
        {detail && <p className="text-xs text-slate-500 mt-0.5">{detail}</p>}
        {children}
      </div>
    </div>
  );
}

// ── Model list ────────────────────────────────────────────────────────────

function ModelList({ models, onDownload }: { models: ModelInfo[]; onDownload: () => void }) {
  const hasCached = models.some((m) => m.cached);

  return (
    <div className="mt-2 space-y-1">
      {models.map((m) => (
        <div key={m.name} className="flex items-center gap-2 text-xs">
          <span className={m.cached ? 'text-green-600' : 'text-slate-400'}>
            {m.cached ? '✓' : '○'}
          </span>
          <span className="font-mono text-slate-700">{m.name}</span>
          {m.cached && m.size_mb != null && (
            <span className="text-slate-400">{m.size_mb.toFixed(0)} MB</span>
          )}
        </div>
      ))}
      {!hasCached && (
        <button
          onClick={onDownload}
          className="mt-2 text-xs px-3 py-1.5 bg-indigo-600 text-white rounded font-medium hover:bg-indigo-700 transition-colors"
        >
          Download Whisper large-v3 (~3 GB)
        </button>
      )}
    </div>
  );
}

// ── Token input ───────────────────────────────────────────────────────────

function TokenInput({ onSaved }: { onSaved: () => void }) {
  const [token, setToken] = useState('');
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleSave() {
    const t = token.trim();
    if (!t) return;
    setSaving(true);
    setResult(null);
    try {
      const res = await saveToken(t);
      if (res.saved && res.valid) {
        setResult({ ok: true, msg: 'Token saved and validated. Speaker diarization is enabled.' });
        setToken('');
        setTimeout(onSaved, 1200);
      } else if (res.saved && !res.valid) {
        setResult({ ok: false, msg: res.error ?? 'Token saved but could not be validated. Check HuggingFace access.' });
      } else {
        setResult({ ok: false, msg: res.error ?? 'Failed to save token.' });
      }
    } catch {
      setResult({ ok: false, msg: 'Network error. Is the API running?' });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="mt-3 space-y-2">
      <input
        ref={inputRef}
        type="password"
        value={token}
        onChange={(e) => setToken(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && handleSave()}
        placeholder="hf_xxxxxxxxxxxxxxxx"
        className="w-full font-mono text-sm px-3 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
        autoComplete="off"
        spellCheck={false}
      />
      <button
        onClick={handleSave}
        disabled={saving || !token.trim()}
        className="text-sm px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center gap-2"
      >
        {saving && <IconSpinner />}
        {saving ? 'Validating…' : 'Save Token'}
      </button>
      {result && (
        <p className={`text-xs px-3 py-2 rounded-lg ${result.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
          {result.msg}
        </p>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

export default function SetupPage() {
  const [status, setStatus] = useState<SetupCheckResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [showTokenInput, setShowTokenInput] = useState(false);

  function load() {
    setLoading(true);
    checkSetup()
      .then((s) => { setStatus(s); setApiError(null); })
      .catch(() => setApiError('Cannot connect to Transcript Engine API. Is it running on port 9097?'))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    checkSetup()
      .then((s) => { setStatus(s); setApiError(null); })
      .catch(() => setApiError('Cannot connect to Transcript Engine API. Is it running on port 9097?'))
      .finally(() => setLoading(false));
  }, []);

  async function handleDownload() {
    setDownloading(true);
    await triggerModelDownload();
    // Poll until cached
    const interval = setInterval(() => {
      checkSetup().then((s) => {
        setStatus(s);
        if (s.whisper_models.some((m) => m.cached)) {
          clearInterval(interval);
          setDownloading(false);
        }
      }).catch(() => {});
    }, 5000);
    // Stop polling after 20 minutes
    setTimeout(() => { clearInterval(interval); setDownloading(false); }, 1200000);
  }

  const hf = status?.hf_token;
  const whisperCached = status?.whisper_models.some((m) => m.cached) ?? false;

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <div className="max-w-2xl mx-auto flex items-center gap-3">
          <Link href="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <span className="font-semibold text-slate-900">Transcript Engine</span>
          </Link>
          <span className="text-slate-300 mx-1">·</span>
          <span className="text-slate-500 text-sm">Setup</span>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-10">
        <div className="mb-8 flex items-start justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 mb-1">System Setup</h1>
            <p className="text-slate-500 text-sm">Everything runs locally — no data leaves your machine.</p>
          </div>
          <button
            onClick={load}
            className="text-sm text-indigo-600 hover:text-indigo-800 font-medium transition-colors"
          >
            Refresh
          </button>
        </div>

        {loading && (
          <div className="flex items-center gap-3 text-slate-500 py-8">
            <IconSpinner />
            Checking system…
          </div>
        )}

        {apiError && (
          <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm mb-6">
            {apiError}
          </div>
        )}

        {status && (
          <div className="space-y-6">

            {/* Core requirements */}
            <Card className="p-4 bg-white shadow-sm border-slate-200">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Core Requirements</h2>

              <StatusRow
                status="ok"
                label={`Python ${status.python_version}`}
                detail="Python runtime"
              />

              <StatusRow
                status={status.ffmpeg ? 'ok' : 'error'}
                label={status.ffmpeg ? `FFmpeg${status.ffmpeg_version ? ` ${status.ffmpeg_version}` : ''}` : 'FFmpeg — Not found'}
                detail={status.ffmpeg
                  ? 'Audio conversion available'
                  : 'Install: brew install ffmpeg (macOS) · apt install ffmpeg (Linux)'}
              />

              <StatusRow
                status={status.output_dir_writable ? 'ok' : 'error'}
                label={status.output_dir_writable ? 'Output directory writable' : 'Output directory not writable'}
                detail="Transcript artifacts are written to ./outputs/"
              />
            </Card>

            {/* Whisper model */}
            <Card className="p-4 bg-white shadow-sm border-slate-200">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Whisper Model</h2>

              <StatusRow
                status={
                  downloading ? 'loading' :
                  whisperCached ? 'ok' :
                  'warn'
                }
                label={
                  downloading ? 'Downloading Whisper large-v3…' :
                  whisperCached ? 'Whisper model cached' :
                  'Whisper model not downloaded'
                }
                detail={
                  downloading ? 'This may take several minutes. Do not close this window.' :
                  whisperCached ? 'Ready for transcription' :
                  'The model will be downloaded automatically on first transcription (~3 GB)'
                }
              >
                <ModelList
                  models={status.whisper_models}
                  onDownload={handleDownload}
                />
                {status.alignment_models.length > 0 && (
                  <div className="mt-3">
                    <p className="text-xs text-slate-400 mb-1">Alignment models:</p>
                    {status.alignment_models.map((m) => (
                      <div key={m.name} className="flex items-center gap-2 text-xs">
                        <span className="text-green-600">✓</span>
                        <span className="font-mono text-slate-600">{m.name}</span>
                        {m.size_mb != null && <span className="text-slate-400">{m.size_mb.toFixed(0)} MB</span>}
                      </div>
                    ))}
                  </div>
                )}
              </StatusRow>
            </Card>

            {/* HuggingFace Token */}
            <Card className="p-4 bg-white shadow-sm border-slate-200">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                Speaker Diarization (HuggingFace)
              </h2>

              {!hf?.present && (
                <StatusRow
                  status="warn"
                  label="HuggingFace token not configured"
                  detail="Without a token, all speech is attributed to a single speaker. Transcription still works."
                >
                  {!showTokenInput ? (
                    <button
                      onClick={() => setShowTokenInput(true)}
                      className="mt-2 text-xs text-indigo-600 hover:text-indigo-800 font-medium"
                    >
                      + Add token
                    </button>
                  ) : (
                    <TokenInput onSaved={load} />
                  )}
                </StatusRow>
              )}

              {hf?.present && hf.valid === false && (
                <StatusRow
                  status="error"
                  label="Token invalid or expired"
                  detail={hf.error ?? 'Generate a new token at huggingface.co/settings/tokens'}
                >
                  <TokenInput onSaved={load} />
                </StatusRow>
              )}

              {hf?.present && hf.valid === true && (
                <>
                  <StatusRow
                    status="ok"
                    label="HuggingFace token valid"
                    detail="Token authenticated successfully"
                  />
                  <StatusRow
                    status={hf.diarization_access ? 'ok' : 'error'}
                    label={hf.diarization_access
                      ? 'pyannote/speaker-diarization-3.1 — Access granted'
                      : 'pyannote/speaker-diarization-3.1 — No access'}
                    detail={hf.diarization_access
                      ? undefined
                      : hf.error ?? 'Accept license at huggingface.co/pyannote/speaker-diarization-3.1'}
                  />
                  <StatusRow
                    status={hf.segmentation_access ? 'ok' : 'error'}
                    label={hf.segmentation_access
                      ? 'pyannote/segmentation-3.0 — Access granted'
                      : 'pyannote/segmentation-3.0 — No access'}
                    detail={!hf.segmentation_access
                      ? 'Accept license at huggingface.co/pyannote/segmentation-3.0'
                      : undefined}
                  />
                </>
              )}

              {hf?.present && hf.valid === null && (
                <StatusRow
                  status="warn"
                  label="Token present but not validated"
                  detail={hf.error ?? 'Could not reach HuggingFace to validate'}
                />
              )}

              {!hf?.present && (
                <div className="mt-3 p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <p className="text-xs font-medium text-slate-700 mb-2">How to get a token:</p>
                  <ol className="text-xs text-slate-600 space-y-1 list-decimal list-inside">
                    <li>Create a free account at <span className="font-medium">huggingface.co</span></li>
                    <li>Generate a token at <code className="bg-slate-100 px-1 rounded">huggingface.co/settings/tokens</code></li>
                    <li>Accept license at <code className="bg-slate-100 px-1 rounded">huggingface.co/pyannote/speaker-diarization-3.1</code></li>
                    <li>Accept license at <code className="bg-slate-100 px-1 rounded">huggingface.co/pyannote/segmentation-3.0</code></li>
                  </ol>
                </div>
              )}
            </Card>

            {/* Profiles */}
            <Card className="p-4 bg-white shadow-sm border-slate-200">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Vocabulary Profiles</h2>
              <StatusRow
                status={status.profiles.length > 0 ? 'ok' : 'warn'}
                label={`${status.profiles.length} profile${status.profiles.length !== 1 ? 's' : ''} available`}
                detail={status.profiles.join(', ')}
              />
            </Card>

            {/* Overall status */}
            <div className={`p-4 rounded-lg border ${status.ready ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
              <div className="flex items-center gap-2 mb-1">
                {status.ready ? <IconCheck /> : <IconWarn />}
                <p className={`text-sm font-semibold ${status.ready ? 'text-green-800' : 'text-amber-800'}`}>
                  {status.ready ? 'Ready to transcribe' : 'Setup incomplete'}
                </p>
              </div>
              <p className={`text-xs ${status.ready ? 'text-green-700' : 'text-amber-700'}`}>
                {status.ready
                  ? 'All requirements met. Upload a recording to get started.'
                  : (!status.ffmpeg
                      ? 'FFmpeg is required. Install it to continue.'
                      : !whisperCached
                      ? 'Whisper model will be downloaded automatically on first transcription.'
                      : 'Check the issues above.')}
              </p>
            </div>

            <Link
              href="/"
              className="block w-full text-center px-4 py-3 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
            >
              {status.ready ? 'Start Transcribing →' : 'Continue anyway →'}
            </Link>
          </div>
        )}
      </main>
    </div>
  );
}
