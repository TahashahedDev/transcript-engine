/**
 * PRD diagrams.
 *
 * Built as HTML/CSS rather than an image or a chart library: they must stay
 * legible when printed to PDF, reflow on a phone, and be readable by a screen
 * reader. Every diagram here encodes a real relationship from the repository —
 * none is decorative.
 */

import type { ReactNode } from 'react';

// ── shared primitives ───────────────────────────────────────────────────────

function Node({
  label,
  sub,
  tone = 'default',
}: {
  label: string;
  sub?: string;
  tone?: 'default' | 'accent' | 'muted' | 'optional';
}) {
  const tones = {
    default: 'border-stone-300 bg-white',
    accent: 'border-teal-700/40 bg-teal-50',
    muted: 'border-stone-200 bg-stone-100',
    optional: 'border-dashed border-amber-500/50 bg-amber-50',
  } as const;
  return (
    <div className={`min-w-0 rounded-sm border px-3 py-2 ${tones[tone]}`}>
      <div className="text-[13px] leading-tight font-medium text-stone-800">{label}</div>
      {sub && <div className="mt-0.5 text-[11px] leading-tight text-stone-500">{sub}</div>}
    </div>
  );
}

function Arrow({ vertical = false }: { vertical?: boolean }) {
  return (
    <div
      aria-hidden="true"
      className={
        vertical
          ? 'ml-5 h-3.5 w-px bg-stone-300'
          : 'hidden h-px w-4 shrink-0 self-center bg-stone-300 sm:block'
      }
    />
  );
}

export function Figure({
  n,
  title,
  note,
  children,
}: {
  n: string;
  title: string;
  note?: string;
  children: ReactNode;
}) {
  return (
    <figure className="my-6 break-inside-avoid">
      <figcaption className="mb-3 flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="font-mono text-[11px] tracking-wide text-teal-800">FIG {n}</span>
        <span className="text-[13px] font-medium text-stone-700">{title}</span>
      </figcaption>
      <div className="overflow-x-auto rounded-sm border border-stone-200 bg-stone-50/60 p-4 sm:p-5">
        {children}
      </div>
      {note && <p className="mt-2 text-[12px] leading-relaxed text-stone-500">{note}</p>}
    </figure>
  );
}

// ── 02 · product pipeline ───────────────────────────────────────────────────

export function PipelineDiagram() {
  const stages = [
    { label: 'Upload', sub: 'audio / video' },
    { label: 'Preprocess', sub: 'ffmpeg · 16 kHz mono' },
    { label: 'Speech-to-text', sub: 'Parakeet TDT · GPU' },
    { label: 'Diarization', sub: 'pyannote · parallel' },
    { label: 'Merge', sub: 'words ↔ speakers' },
    { label: 'Processors', sub: 'vocabulary · cleanup' },
    { label: 'Artifacts', sub: '5 formats + bundle' },
  ];
  return (
    <div className="flex min-w-[640px] items-stretch gap-1">
      {stages.map((s, i) => (
        <div key={s.label} className="flex flex-1 items-stretch gap-1">
          <div className="flex-1">
            <Node
              label={s.label}
              sub={s.sub}
              tone={i === 2 || i === 3 ? 'accent' : 'default'}
            />
          </div>
          {i < stages.length - 1 && <Arrow />}
        </div>
      ))}
    </div>
  );
}

// ── 06 · user journey ───────────────────────────────────────────────────────

const JOURNEY = [
  {
    stage: 'Upload',
    action: 'Drops a recording onto the page',
    system: 'Validates format and size against the server limit, streams to disk',
    friction: 'Large files over a slow uplink have no progress indicator',
  },
  {
    stage: 'Configure',
    action: 'Picks profile, timestamp mode, quality mode',
    system: 'Offers only the modes the running backend supports',
    friction: 'A mode can advertise a feature that is not configured (now disclosed, UX-002)',
  },
  {
    stage: 'Submit',
    action: 'Starts the job',
    system: 'Returns a job ID immediately, before any GPU work begins',
    friction: 'None verified',
  },
  {
    stage: 'Monitor',
    action: 'Watches live progress',
    system: 'Streams stage-level events over SSE with reconnection',
    friction: 'A wedged job takes up to 45 minutes to reach a terminal state',
  },
  {
    stage: 'Review',
    action: 'Reads transcript and meeting artifacts',
    system: 'Renders transcript, summary, action items, decisions, timeline',
    friction: 'No accuracy signal — the user cannot tell how much to trust it',
  },
  {
    stage: 'Export',
    action: 'Downloads individual files or the bundle',
    system: 'Serves artifacts; TTL cleanup removes them later',
    friction: 'A restart loses the job entirely before download (FR-006)',
  },
];

export function JourneyDiagram() {
  return (
    <div className="min-w-[560px]">
      <div className="mb-2 grid grid-cols-[110px_1fr_1fr_1fr] gap-3 border-b border-stone-200 pb-2">
        {['Stage', 'User action', 'System response', 'Friction'].map((h) => (
          <div key={h} className="text-[10px] font-semibold tracking-wider text-stone-500 uppercase">
            {h}
          </div>
        ))}
      </div>
      <ol className="divide-y divide-stone-200">
        {JOURNEY.map((j, i) => (
          <li key={j.stage} className="grid grid-cols-[110px_1fr_1fr_1fr] gap-3 py-2.5">
            <div className="flex items-start gap-2">
              <span className="mt-0.5 font-mono text-[10px] text-stone-400">{i + 1}</span>
              <span className="text-[13px] font-medium text-stone-800">{j.stage}</span>
            </div>
            <div className="text-[12px] leading-snug text-stone-600">{j.action}</div>
            <div className="text-[12px] leading-snug text-stone-600">{j.system}</div>
            <div
              className={`text-[12px] leading-snug ${
                j.friction === 'None verified' ? 'text-stone-400' : 'text-amber-800'
              }`}
            >
              {j.friction}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

// ── 07 · capability map ─────────────────────────────────────────────────────

const CAPABILITIES = [
  {
    group: 'Core transcription',
    items: [
      { name: 'Upload & validation', state: 'shipped' },
      { name: 'Audio preprocessing', state: 'shipped' },
      { name: 'ASR — Parakeet / WhisperX', state: 'shipped' },
      { name: 'Speaker diarization', state: 'shipped' },
      { name: 'Live progress (SSE)', state: 'shipped' },
      { name: 'Export — 5 formats + bundle', state: 'shipped' },
    ],
  },
  {
    group: 'AI enhancement',
    items: [
      { name: 'Vocabulary profiles', state: 'shipped' },
      { name: 'Rule-based meeting intelligence', state: 'shipped' },
      { name: 'AI grammar pass (opt-in)', state: 'shipped' },
      { name: 'Local inference — Ollama', state: 'shipped' },
      { name: 'Output fidelity checks', state: 'gap' },
      { name: 'Accuracy evaluation', state: 'gap' },
    ],
  },
  {
    group: 'Platform',
    items: [
      { name: 'FastAPI job API', state: 'shipped' },
      { name: 'Next.js UI + same-origin proxy', state: 'shipped' },
      { name: 'GPU/VRAM-aware scheduling', state: 'shipped' },
      { name: 'Filesystem artifact storage', state: 'shipped' },
      { name: 'Durable job store', state: 'planned' },
      { name: 'Cloud API (/api/v2)', state: 'dormant' },
    ],
  },
];

const CAP_STATE: Record<string, { dot: string; label: string }> = {
  shipped: { dot: 'bg-teal-700', label: 'Implemented' },
  gap: { dot: 'bg-amber-500', label: 'Gap' },
  planned: { dot: 'bg-stone-400', label: 'Planned' },
  dormant: { dot: 'bg-stone-300 ring-1 ring-stone-400', label: 'Dormant' },
};

export function CapabilityMap() {
  return (
    <div>
      <div className="grid gap-4 sm:grid-cols-3">
        {CAPABILITIES.map((c) => (
          <div key={c.group}>
            <h4 className="mb-2 border-b border-stone-300 pb-1.5 text-[11px] font-semibold tracking-wider text-stone-600 uppercase">
              {c.group}
            </h4>
            <ul className="space-y-1.5">
              {c.items.map((i) => (
                <li key={i.name} className="flex items-start gap-2">
                  <span
                    aria-hidden="true"
                    className={`mt-1.5 size-1.5 shrink-0 rounded-full ${CAP_STATE[i.state].dot}`}
                  />
                  <span className="text-[12.5px] leading-snug text-stone-700">
                    {i.name}
                    <span className="sr-only"> — {CAP_STATE[i.state].label}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 border-t border-stone-200 pt-3">
        {Object.entries(CAP_STATE).map(([k, v]) => (
          <span key={k} className="flex items-center gap-1.5 text-[11px] text-stone-500">
            <span aria-hidden="true" className={`size-1.5 rounded-full ${v.dot}`} />
            {v.label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── 09 · AI pipeline ────────────────────────────────────────────────────────

export function AiDiagram() {
  // A single vertical sequence rather than a grid: the flow is strictly linear,
  // and a two-column layout invites the reader to scan one column top-to-bottom
  // and miss the actual order. The band labels carry the real information —
  // where the deterministic guarantee ends and generative risk begins.
  const core = [
    { label: 'Audio', sub: 'preprocessed · 16 kHz mono', tone: 'muted' as const },
    {
      label: 'Parakeet TDT / WhisperX',
      sub: 'speech-to-text · transformation, not generative',
      tone: 'accent' as const,
    },
    {
      label: 'pyannote diarization',
      sub: 'speaker clustering · not generative',
      tone: 'accent' as const,
    },
    { label: 'Speaker-labelled transcript', sub: 'word-level timestamps', tone: 'default' as const },
  ];

  return (
    <div className="mx-auto max-w-md">
      <div className="mb-2 text-[10px] font-semibold tracking-wider text-teal-800 uppercase">
        Deterministic core — always runs
      </div>
      <div className="border-l-2 border-teal-700/30 pl-4">
        {core.map((n, i) => (
          <div key={n.label}>
            <Node label={n.label} sub={n.sub} tone={n.tone} />
            {i < core.length - 1 && <Arrow vertical />}
          </div>
        ))}
      </div>

      <div className="py-1 pl-4">
        <Arrow vertical />
      </div>

      <div className="mb-2 text-[10px] font-semibold tracking-wider text-amber-700 uppercase">
        Generative layer — optional, opt-in, must degrade safely
      </div>
      <div className="border-l-2 border-dashed border-amber-500/40 pl-4">
        <Node label="AI grammar pass" sub="LLM · constrained by prompt only" tone="optional" />
        <Arrow vertical />
        <Node label="Final artifacts" sub="transcript + 5 meeting documents" />
      </div>
    </div>
  );
}

// ── 10 · system architecture ────────────────────────────────────────────────

export function SystemDiagram() {
  return (
    <div className="min-w-[520px] space-y-2">
      <div className="rounded-sm border border-stone-300 bg-white p-3">
        <div className="mb-2 text-[10px] font-semibold tracking-wider text-stone-500 uppercase">
          Public boundary — one port only
        </div>
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
          <Node label="Browser" sub="the only external client" />
          <span aria-hidden="true" className="text-stone-400">
            →
          </span>
          <Node label="Next.js :9098" sub="UI + same-origin /api proxy" tone="accent" />
        </div>
      </div>

      <div className="text-center text-[11px] text-stone-500">
        <span aria-hidden="true">↓</span> server-side only · 127.0.0.1
      </div>

      <div className="rounded-sm border border-stone-300 bg-white p-3">
        <div className="mb-2 text-[10px] font-semibold tracking-wider text-stone-500 uppercase">
          Loopback — never externally exposed
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <Node label="FastAPI :9097" sub="jobs · progress · artifacts" tone="accent" />
          <Node label="Pipeline orchestrator" sub="async, single worker" />
          <Node label="Ollama :11434" sub="optional grammar LLM" tone="optional" />
        </div>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          <Node label="GPU layer" sub="Parakeet + pyannote, VRAM-aware lock" tone="accent" />
          <Node label="Filesystem" sub="temp/ + outputs/, TTL cleanup" />
          <Node label="PostgreSQL" sub="schema exists — not wired" tone="muted" />
        </div>
      </div>
    </div>
  );
}
