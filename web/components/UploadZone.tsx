'use client';

import { useRef, useState } from 'react';

const ACCEPTED = '.aac,.mp3,.wav,.m4a,.flac,.mp4,.mov,.ogg';
const ACCEPTED_SET = new Set(['.aac', '.mp3', '.wav', '.m4a', '.flac', '.mp4', '.mov', '.ogg']);

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Fallback only. The real limit comes from the backend (APIConfig.max_upload_mb,
// surfaced via /api/setup/check) so the two can't drift — the client previously
// hardcoded 500 MB while the server accepted 2000 MB, rejecting files locally
// that would have uploaded fine.
const DEFAULT_MAX_UPLOAD_MB = 2000;

interface Props {
  onFile: (file: File | null) => void;
  maxUploadMb?: number;
}

export function UploadZone({ onFile, maxUploadMb = DEFAULT_MAX_UPLOAD_MB }: Props) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function validate(f: File): boolean {
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!ACCEPTED_SET.has(ext)) {
      setError(`Unsupported format: ${ext}. Accepted: AAC, MP3, WAV, M4A, FLAC, MP4, MOV, OGG`);
      return false;
    }
    if (f.size > maxUploadMb * 1024 * 1024) {
      setError(`File too large. Maximum ${maxUploadMb} MB.`);
      return false;
    }
    return true;
  }

  function handleFile(f: File) {
    setError(null);
    if (!validate(f)) return;
    setFile(f);
    onFile(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
    // Reset so picking the *same* file again still fires a change event.
    e.target.value = '';
  }

  function clearFile() {
    setFile(null);
    setError(null);
    onFile(null);
  }

  return (
    <div>
      {/*
        A real <button> rather than a click-handling div: the previous version
        could not be reached or activated by keyboard at all, so the file
        picker was unusable without a mouse.
      */}
      <button
        type="button"
        aria-label={file ? `Selected ${file.name}. Choose a different file` : 'Choose an audio or video file to transcribe'}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex min-h-55 w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-10 transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 ${
          dragging
            ? 'border-indigo-500 bg-indigo-50'
            : file
              ? 'border-indigo-300 bg-indigo-50/40 hover:bg-indigo-50/70'
              : 'border-slate-300 bg-white hover:border-indigo-400 hover:bg-slate-50'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={ACCEPTED}
          onChange={onInputChange}
          tabIndex={-1}
        />

        {file ? (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-indigo-100">
              <svg aria-hidden="true" className="h-7 w-7 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <div className="max-w-full text-center">
              <p className="truncate px-2 font-semibold text-slate-800">{file.name}</p>
              <p className="mt-0.5 text-sm tabular-nums text-slate-500">{formatBytes(file.size)}</p>
            </div>
            <span className="text-xs text-indigo-600">Click or drop to replace</span>
          </>
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-slate-100">
              <svg aria-hidden="true" className="h-7 w-7 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="text-center">
              <p className="font-medium text-slate-700">Drop your audio file here</p>
              <p className="mt-1 text-sm text-slate-400">or click to browse</p>
            </div>
            <p className="px-4 text-center text-xs text-slate-400">
              AAC · MP3 · WAV · M4A · FLAC · MP4 · MOV · OGG · up to {maxUploadMb} MB
            </p>
          </>
        )}
      </button>

      {file && (
        <button
          type="button"
          onClick={clearFile}
          className="mt-2 cursor-pointer text-xs text-slate-500 underline-offset-2 transition-colors hover:text-slate-700 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
        >
          Remove file
        </button>
      )}

      {error && (
        <p role="alert" className="mt-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}
