'use client';

/**
 * Transcript Engine — Product Requirements Document.
 *
 * A document first, an interface second. It lives on its own route and shares
 * nothing with the transcription workflow: no API calls, no app state, no
 * imports from lib/. Deleting this directory would leave the product
 * untouched.
 *
 * Interactions earn their place or they aren't here — scroll-spy navigation,
 * requirement filtering, progressive disclosure on requirement detail, and an
 * executive view that hides the deep sections. Everything degrades to a
 * readable document when printed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  DECISIONS,
  KPIS,
  PRIORITY_LABEL,
  REQUIREMENTS,
  RISKS,
  STATUS_LABEL,
  type Category,
  type Priority,
  type Requirement,
  type Status,
} from './data';
import {
  AiDiagram,
  CapabilityMap,
  Figure,
  JourneyDiagram,
  PipelineDiagram,
  SystemDiagram,
} from './diagrams';

// ── section registry ────────────────────────────────────────────────────────

interface Section {
  id: string;
  n: string;
  nav: string;
  title: string;
  /** Shown in executive view. The rest are detail sections. */
  exec?: boolean;
}

const SECTIONS: Section[] = [
  { id: 'overview', n: '01', nav: 'Overview', title: 'Executive overview', exec: true },
  { id: 'product', n: '02', nav: 'Product', title: 'The product in one view', exec: true },
  { id: 'problem', n: '03', nav: 'Problem', title: 'Problem, opportunity, vision', exec: true },
  { id: 'framework', n: '04', nav: 'Framework', title: 'Why this PRD structure' },
  { id: 'users', n: '05', nav: 'Users', title: 'Users and core use cases' },
  { id: 'journey', n: '06', nav: 'Journey', title: 'User journey' },
  { id: 'capabilities', n: '07', nav: 'Capabilities', title: 'Product capability map', exec: true },
  { id: 'requirements', n: '08', nav: 'Requirements', title: 'Requirements' },
  { id: 'ai', n: '09', nav: 'AI/ML', title: 'AI/ML product design' },
  { id: 'architecture', n: '10', nav: 'Architecture', title: 'Technical architecture' },
  { id: 'performance', n: '11', nav: 'Performance', title: 'Performance and reliability' },
  { id: 'metrics', n: '12', nav: 'Metrics', title: 'Success metrics' },
  { id: 'roadmap', n: '13', nav: 'Roadmap', title: 'Scope and roadmap', exec: true },
  { id: 'risks', n: '14', nav: 'Risks', title: 'Risks and dependencies', exec: true },
  { id: 'decisions', n: '15', nav: 'Decisions', title: 'Decisions and open questions', exec: true },
  { id: 'takeaway', n: '16', nav: 'Takeaway', title: 'Executive takeaway', exec: true },
];

// ── evidence labels ─────────────────────────────────────────────────────────

const STATUS_STYLE: Record<Status, string> = {
  verified: 'bg-teal-50 text-teal-900 ring-teal-700/25',
  partial: 'bg-sky-50 text-sky-900 ring-sky-700/25',
  gap: 'bg-amber-50 text-amber-900 ring-amber-700/30',
  planned: 'bg-stone-100 text-stone-700 ring-stone-400/40',
  gated: 'bg-rose-50 text-rose-900 ring-rose-700/25',
  unverified: 'bg-stone-100 text-stone-600 ring-stone-400/40',
};

const PRIORITY_STYLE: Record<Priority, string> = {
  must: 'text-stone-900 font-semibold',
  should: 'text-stone-700',
  could: 'text-stone-500',
  wont: 'text-stone-400 line-through',
};

function Chip({ status }: { status: Status }) {
  return (
    <span
      className={`inline-block rounded-sm px-1.5 py-0.5 text-[10px] font-semibold tracking-wide whitespace-nowrap uppercase ring-1 ring-inset ${STATUS_STYLE[status]}`}
    >
      {STATUS_LABEL[status]}
    </span>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[10px] font-semibold tracking-wider text-stone-500 uppercase">
      {children}
    </span>
  );
}

// ── section shell ───────────────────────────────────────────────────────────

function SectionHeading({ s }: { s: Section }) {
  return (
    <header className="mb-5 border-b border-stone-300 pb-3">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-[12px] text-teal-800 tabular-nums">{s.n}</span>
        <h2 className="prd-display text-[26px] leading-tight font-semibold text-stone-900">
          {s.title}
        </h2>
      </div>
    </header>
  );
}

// ── requirements table ──────────────────────────────────────────────────────

const CATEGORIES: Category[] = [
  'Functional',
  'UX',
  'AI/ML',
  'Platform',
  'Performance',
  'Security',
  'Reliability',
  'Architecture',
];

function RequirementRow({ r }: { r: Requirement }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="border-b border-stone-200 last:border-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="grid w-full cursor-pointer grid-cols-[64px_1fr_auto] items-start gap-3 py-2.5 text-left hover:bg-stone-50 sm:grid-cols-[72px_1fr_88px_128px]"
      >
        <span className="prd-id font-mono text-[11px] font-semibold text-teal-800 tabular-nums">
          {r.id}
        </span>
        <span className="text-[13.5px] leading-snug text-stone-800">{r.requirement}</span>
        <span className={`hidden text-[12px] sm:block ${PRIORITY_STYLE[r.priority]}`}>
          {PRIORITY_LABEL[r.priority]}
        </span>
        <span className="flex items-center gap-2">
          <Chip status={r.status} />
          <span
            aria-hidden="true"
            className={`prd-caret text-stone-400 transition-transform ${open ? 'rotate-90' : ''}`}
          >
            ›
          </span>
        </span>
      </button>
      <div className={`prd-detail ${open ? 'block' : 'hidden'} pb-4 pl-0 sm:pl-[84px]`}>
        <dl className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
          <div>
            <dt>
              <Label>Rationale</Label>
            </dt>
            <dd className="mt-0.5 text-[12.5px] leading-snug text-stone-600">{r.rationale}</dd>
          </div>
          <div>
            <dt>
              <Label>Acceptance criteria</Label>
            </dt>
            <dd className="mt-0.5 text-[12.5px] leading-snug text-stone-600">{r.acceptance}</dd>
          </div>
          <div className="sm:col-span-2">
            <dt>
              <Label>Evidence</Label>
            </dt>
            <dd className="mt-0.5 font-mono text-[11.5px] leading-snug text-stone-500">
              {r.evidence}
            </dd>
          </div>
        </dl>
      </div>
    </li>
  );
}

function RequirementsBrowser() {
  const [cat, setCat] = useState<Category | 'All'>('All');
  const [onlyGaps, setOnlyGaps] = useState(false);

  const shown = useMemo(
    () =>
      REQUIREMENTS.filter(
        (r) =>
          (cat === 'All' || r.category === cat) &&
          (!onlyGaps || r.status === 'gap' || r.status === 'partial' || r.status === 'gated'),
      ),
    [cat, onlyGaps],
  );

  const counts = useMemo(() => {
    const verified = REQUIREMENTS.filter((r) => r.status === 'verified').length;
    const gaps = REQUIREMENTS.filter((r) => r.status === 'gap').length;
    return { total: REQUIREMENTS.length, verified, gaps };
  }, []);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[12px] text-stone-600">
        <span>
          <strong className="text-stone-900 tabular-nums">{counts.total}</strong> requirements
        </span>
        <span>
          <strong className="text-teal-800 tabular-nums">{counts.verified}</strong> already
          implemented
        </span>
        <span>
          <strong className="text-amber-700 tabular-nums">{counts.gaps}</strong> open gaps
        </span>
      </div>

      <div className="prd-controls mb-4 flex flex-wrap items-center gap-1.5">
        {(['All', ...CATEGORIES] as const).map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => setCat(c)}
            aria-pressed={cat === c}
            className={`cursor-pointer rounded-sm px-2.5 py-1 text-[12px] transition-colors ${
              cat === c
                ? 'bg-stone-900 text-white'
                : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
            }`}
          >
            {c}
          </button>
        ))}
        <span aria-hidden="true" className="mx-1 h-4 w-px bg-stone-300" />
        <button
          type="button"
          onClick={() => setOnlyGaps((v) => !v)}
          aria-pressed={onlyGaps}
          className={`cursor-pointer rounded-sm px-2.5 py-1 text-[12px] transition-colors ${
            onlyGaps ? 'bg-amber-600 text-white' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
          }`}
        >
          Unfinished only
        </button>
      </div>

      <div className="mb-1.5 hidden grid-cols-[72px_1fr_88px_128px] gap-3 border-b border-stone-300 pb-1.5 sm:grid">
        <Label>ID</Label>
        <Label>Requirement</Label>
        <Label>Priority</Label>
        <Label>Status</Label>
      </div>

      {shown.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-stone-500">
          No requirements match this filter.
        </p>
      ) : (
        <ul>
          {shown.map((r) => (
            <RequirementRow key={r.id} r={r} />
          ))}
        </ul>
      )}
      <p className="mt-3 text-[11.5px] text-stone-500">
        Select a requirement to see its rationale, acceptance criteria and source evidence.
      </p>
    </div>
  );
}

// ── page ────────────────────────────────────────────────────────────────────

export default function PrdPage() {
  const [active, setActive] = useState('overview');
  const [execOnly, setExecOnly] = useState(false);
  const [progress, setProgress] = useState(0);
  const navRef = useRef<HTMLDivElement>(null);

  const visible = useMemo(() => (execOnly ? SECTIONS.filter((s) => s.exec) : SECTIONS), [execOnly]);

  // Scroll-spy and reading progress share one listener.
  //
  // Deliberately not an IntersectionObserver: with sections this tall, several
  // intersect the viewport at once and picking the topmost one keeps a section
  // highlighted long after the reader has scrolled past it. "The last section
  // whose heading has crossed the nav" is what a reader actually means by
  // "where am I", so it is computed directly.
  useEffect(() => {
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      setProgress(max > 0 ? Math.min(1, h.scrollTop / max) : 0);

      const line = 120; // just below the sticky nav
      let current = visible[0]?.id ?? '';
      for (const s of visible) {
        const el = document.getElementById(s.id);
        if (el && el.getBoundingClientRect().top <= line) current = s.id;
      }
      // At the very bottom the final section may never cross the line.
      if (max > 0 && max - h.scrollTop < 4) current = visible[visible.length - 1]?.id ?? current;
      setActive(current);
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, [visible]);

  // Keep the active nav pill in view on narrow screens, where the nav scrolls
  // horizontally and the current section can sit off-screen.
  useEffect(() => {
    const el = navRef.current?.querySelector<HTMLElement>(`[data-nav="${active}"]`);
    el?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [active]);

  const jump = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  return (
    <div className="prd-root min-h-screen bg-[#faf9f6] text-stone-800">
      <style>{PRD_CSS}</style>

      {/* ── sticky navigation ── */}
      <div className="prd-nav sticky top-0 z-30 border-b border-stone-300 bg-[#faf9f6]/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1180px] items-center gap-4 px-5 py-2.5 sm:px-8">
          <span className="prd-display hidden shrink-0 text-[13px] font-semibold text-stone-900 md:block">
            Transcript Engine
            <span className="ml-2 font-sans text-[11px] font-normal text-stone-500">PRD</span>
          </span>
          {/* The fade mask is the affordance: with 16 sections the list overflows
              even on a wide screen, and a hard clip at the edge reads as a
              layout bug rather than as "there is more this way". */}
          <nav aria-label="Document sections" className="prd-navmask min-w-0 flex-1">
            <div ref={navRef} className="prd-navscroll flex gap-px overflow-x-auto">
              {visible.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  data-nav={s.id}
                  onClick={() => jump(s.id)}
                  aria-current={active === s.id ? 'true' : undefined}
                  className={`shrink-0 cursor-pointer rounded-sm px-1.5 py-1 text-[11.5px] whitespace-nowrap transition-colors ${
                    active === s.id
                      ? 'bg-stone-900 text-white'
                      : 'text-stone-600 hover:bg-stone-200/70'
                  }`}
                >
                  {s.nav}
                </button>
              ))}
            </div>
          </nav>
          <button
            type="button"
            onClick={() => setExecOnly((v) => !v)}
            aria-pressed={execOnly}
            className={`shrink-0 cursor-pointer rounded-sm border px-2.5 py-1 text-[11.5px] transition-colors ${
              execOnly
                ? 'border-teal-800 bg-teal-800 text-white'
                : 'border-stone-300 text-stone-600 hover:bg-stone-100'
            }`}
          >
            Executive view
          </button>
        </div>
        <div
          aria-hidden="true"
          className="h-px origin-left bg-teal-800 transition-transform duration-150"
          style={{ transform: `scaleX(${progress})` }}
        />
      </div>

      <main className="mx-auto max-w-[1180px] px-5 pb-24 sm:px-8">
        {/* ── 01 · executive overview ── */}
        <section id="overview" className="scroll-mt-24 pt-10 sm:pt-14">
          <p className="mb-3 font-mono text-[11px] tracking-widest text-teal-800 uppercase">
            Product Requirements Document
          </p>
          <h1 className="prd-display max-w-[18ch] text-[42px] leading-[1.05] font-semibold text-stone-900 sm:text-[56px]">
            Transcript Engine
          </h1>
          <p className="mt-5 max-w-[62ch] text-[17px] leading-relaxed text-stone-600">
            A self-hosted AI meeting-transcription product. Audio in, speaker-labelled transcript
            and structured meeting notes out — running entirely on infrastructure the operator
            controls, with no recording sent to a third-party service.
          </p>

          <div className="mt-8 grid gap-x-8 gap-y-5 border-y border-stone-300 py-5 sm:grid-cols-3 lg:grid-cols-6">
            {[
              ['Product', 'Meeting transcription engine'],
              ['Users', 'Undefined — Decision 1'],
              ['Core capability', 'Speaker-labelled transcript + meeting artifacts'],
              ['AI stack', 'Parakeet TDT · pyannote · optional local LLM'],
              ['Deployment', 'Self-hosted / rented GPU'],
              ['Status', 'Deployed, hardening'],
            ].map(([k, v]) => (
              <div key={k}>
                <Label>{k}</Label>
                <p className="mt-1 text-[13px] leading-snug text-stone-800">{v}</p>
              </div>
            ))}
          </div>

          <div className="mt-5 flex flex-wrap gap-x-6 gap-y-1 text-[11.5px] text-stone-500">
            <span>
              Version <strong className="text-stone-700">1.0</strong>
            </span>
            <span>
              Source of truth{' '}
              <strong className="font-mono text-stone-700">commit d352874</strong>
            </span>
            <span>
              Status <strong className="text-stone-700">For review</strong>
            </span>
          </div>

          <div className="prd-note mt-6">
            <p>
              <strong>How to read this document.</strong> Every factual claim carries an evidence
              label. <Chip status="verified" /> means confirmed in code.{' '}
              <Chip status="gap" /> is a known shortfall. <Chip status="gated" /> is blocked on a
              stakeholder decision. Nothing here — no metric, benchmark, user count or business
              outcome — has been invented to fill a gap; where a number does not exist, the document
              says so and defines how to obtain it.
            </p>
          </div>
        </section>

        {/* ── 02 · product in one view ── */}
        <section id="product" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[1]} />
          <p className="mb-1 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
            One recording moves through seven stages. Transcription and diarization are the two
            GPU-resident stages and the two that define the product&rsquo;s quality ceiling;
            everything after them is correction and formatting.
          </p>
          <Figure
            n="1"
            title="Processing pipeline"
            note="Diarization runs in parallel with transcription where hardware allows, serialised by a process-wide GPU lock when both would be CUDA-resident at once."
          >
            <PipelineDiagram />
          </Figure>
        </section>

        {/* ── 03 · problem / opportunity / vision ── */}
        <section id="problem" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[2]} />
          <div className="grid gap-5 sm:grid-cols-3">
            {[
              {
                h: 'Problem',
                tag: 'inference' as const,
                b: 'Teams and individuals need meeting transcripts but cannot, or will not, send recordings to a third-party cloud transcription vendor — for confidentiality, regulatory or cost reasons.',
                f: 'Inferred from architecture and README framing. No formal problem statement exists in the repository.',
              },
              {
                h: 'Opportunity',
                tag: 'inference' as const,
                b: 'Modern open ASR models run well on commodity and rented GPUs. A self-hosted pipeline can deliver speaker-labelled transcripts and structured meeting output without a per-minute vendor bill or a data-residency conversation.',
                f: 'Inferred. No market sizing, competitor analysis or pricing comparison exists in the repository.',
              },
              {
                h: 'Vision',
                tag: 'inference' as const,
                b: 'A transcription engine that runs entirely under the operator&rsquo;s control — laptop, workstation or rented GPU — producing transcripts and meeting artifacts with no third-party dependency in the data path.',
                f: 'Synthesised from consistent local-first architectural choices. Should be confirmed or replaced by the product owner.',
              },
            ].map((c) => (
              <div key={c.h} className="border-t-2 border-stone-900 pt-3">
                <h3 className="prd-display mb-2 text-[17px] font-semibold text-stone-900">{c.h}</h3>
                <p
                  className="text-[13.5px] leading-relaxed text-stone-700"
                  dangerouslySetInnerHTML={{ __html: c.b }}
                />
                <p className="mt-3 border-t border-stone-200 pt-2 text-[11.5px] leading-snug text-stone-500">
                  <span className="font-semibold text-sky-800">Inference — </span>
                  {c.f}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* ── 04 · framework ── */}
        {!execOnly && (
          <section id="framework" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[3]} />
            <p className="mb-4 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              &ldquo;PRD&rdquo; is not one document type. Seven patterns are in common industry use
              — none formally standardised, and most teams blend them. The pattern chosen determines
              what the document emphasises and, just as importantly, what it deliberately leaves out.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b border-stone-300">
                    <th className="py-2 pr-4 text-left">
                      <Label>Pattern</Label>
                    </th>
                    <th className="py-2 pr-4 text-left">
                      <Label>Audience</Label>
                    </th>
                    <th className="py-2 pr-4 text-left">
                      <Label>Emphasises</Label>
                    </th>
                    <th className="py-2 text-left">
                      <Label>Fit here</Label>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Business / executive', 'Leadership', 'Market, ROI, strategic fit', 'partial'],
                    ['Product / feature', 'Cross-functional', 'User problem, use cases, scope', 'core'],
                    ['Engineering', 'Engineering', 'Constraints, edge cases', 'partial'],
                    ['Design / UX', 'Design', 'Flows, states, interaction', 'partial'],
                    ['Technical / platform', 'Architects', 'Interfaces, non-functional requirements', 'core'],
                    ['AI / ML', 'Cross-functional + ML', 'Model behaviour, evaluation, failure modes', 'core'],
                    ['Enterprise / formal', 'Compliance', 'Traceability, sign-off', 'no'],
                  ].map(([p, a, e, fit]) => (
                    <tr key={p} className="border-b border-stone-200">
                      <td className="py-2 pr-4 font-medium text-stone-800">{p}</td>
                      <td className="py-2 pr-4 text-stone-600">{a}</td>
                      <td className="py-2 pr-4 text-stone-600">{e}</td>
                      <td className="py-2">
                        {fit === 'core' && (
                          <span className="rounded-sm bg-teal-800 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-white uppercase">
                            Selected
                          </span>
                        )}
                        {fit === 'partial' && (
                          <span className="text-[11px] text-stone-500">Partial</span>
                        )}
                        {fit === 'no' && <span className="text-[11px] text-stone-400">No</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="prd-note mt-5">
              <p className="mb-2">
                <strong>
                  Selected: a hybrid Product / Technical-Platform / AI-ML PRD.
                </strong>{' '}
                Chosen from repository evidence, not preference:
              </p>
              <ul className="ml-4 list-disc space-y-1 text-[13px] text-stone-700">
                <li>
                  The product <strong>already works end to end</strong> — so this documents and
                  hardens a real system rather than specifying a hypothetical one.
                </li>
                <li>
                  Its hardest problems are <strong>operational</strong> (VRAM limits, OOM recovery,
                  stalled jobs) — that demands platform-grade non-functional requirements.
                </li>
                <li>
                  It contains <strong>two distinct AI risk classes</strong> — deterministic
                  transformation and a generative pass — which a generic template has no slot for.
                </li>
                <li>
                  Deployment is <strong>single-tenant and self-hosted</strong>, so formal
                  enterprise traceability would be ceremony without benefit.
                </li>
              </ul>
            </div>
          </section>
        )}

        {/* ── 05 · users ── */}
        {!execOnly && (
          <section id="users" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[4]} />

            <div className="prd-note prd-note-warn mb-5">
              <p>
                <strong>Open question — the most consequential gap in this document.</strong> No user
                research, persona documentation or target-market statement exists in the repository.
                The profiles below describe who the product <em>can currently serve</em>, derived
                from architectural constraints. They are not a validated audience. See Decision 1.
              </p>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              {[
                {
                  h: 'Individual operator',
                  s: 'Supported today',
                  ok: true,
                  b: 'Runs the tool on their own machine or a GPU they rent, transcribes their own recordings. Matches the no-auth, single-worker architecture exactly.',
                },
                {
                  h: 'Small shared team',
                  s: 'Not supported',
                  ok: false,
                  b: 'Several people, one deployment. Plausible from a network-reachable UI, but there is no auth, no per-user isolation, and a shared in-memory job store.',
                },
                {
                  h: 'Domain specialist',
                  s: 'Supported today',
                  ok: true,
                  b: 'Needs correct domain terminology. The banking vocabulary profile shows at least one such case was built for deliberately.',
                },
              ].map((p) => (
                <div key={p.h} className="rounded-sm border border-stone-300 bg-white p-4">
                  <div className="mb-2 flex items-start justify-between gap-2">
                    <h3 className="text-[15px] font-semibold text-stone-900">{p.h}</h3>
                    <span
                      className={`shrink-0 rounded-sm px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase ring-1 ring-inset ${
                        p.ok
                          ? 'bg-teal-50 text-teal-900 ring-teal-700/25'
                          : 'bg-stone-100 text-stone-600 ring-stone-400/40'
                      }`}
                    >
                      {p.s}
                    </span>
                  </div>
                  <p className="text-[13px] leading-relaxed text-stone-600">{p.b}</p>
                </div>
              ))}
            </div>

            <h3 className="prd-display mt-8 mb-3 text-[17px] font-semibold text-stone-900">
              Core use cases
            </h3>
            <ul className="divide-y divide-stone-200 border-y border-stone-200">
              {[
                ['Upload a recorded meeting, get a speaker-labelled transcript', true],
                ['Get summary, action items and decisions without reading the transcript', true],
                ['Get a grammatically cleaned transcript', true],
                ['Transcribe domain-specific audio with correct terminology', true],
                ['Know which named person said something', false],
                ['Review past jobs after a server restart', false],
              ].map(([t, ok]) => (
                <li key={t as string} className="flex items-start gap-3 py-2">
                  <span
                    aria-hidden="true"
                    className={`mt-1 size-1.5 shrink-0 rounded-full ${ok ? 'bg-teal-700' : 'bg-stone-300'}`}
                  />
                  <span className={`text-[13.5px] ${ok ? 'text-stone-700' : 'text-stone-400'}`}>
                    {t as string}
                  </span>
                  <span className="ml-auto shrink-0 text-[11px] text-stone-400">
                    {ok ? 'Supported' : 'Not supported'}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* ── 06 · journey ── */}
        {!execOnly && (
          <section id="journey" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[5]} />
            <Figure
              n="2"
              title="End-to-end user journey with friction points"
              note="Friction column records observed or documented problems only — each maps to a requirement in section 08."
            >
              <JourneyDiagram />
            </Figure>
          </section>
        )}

        {/* ── 07 · capability map ── */}
        <section id="capabilities" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[6]} />
          <p className="mb-1 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
            Most of this product is built. The value of the map is that it shows exactly where it
            is not — and those four items drive the entire next phase of work.
          </p>
          <Figure n="3" title="Capability map by implementation state">
            <CapabilityMap />
          </Figure>
        </section>

        {/* ── 08 · requirements ── */}
        {!execOnly && (
          <section id="requirements" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[7]} />
            <p className="mb-5 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              Requirements are prioritised with MoSCoW against a single Phase&nbsp;1 objective:{' '}
              <strong className="text-stone-800">
                close the self-documented gaps and reach a defensible production baseline
              </strong>{' '}
              — not a growth target, since no business goal is evidenced in the repository.
              Requirements describing working behaviour are recorded so they are not silently
              regressed.
            </p>
            <RequirementsBrowser />
          </section>
        )}

        {/* ── 09 · AI ── */}
        {!execOnly && (
          <section id="ai" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[8]} />
            <p className="mb-1 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              The central AI design decision in this product is already made and worth stating
              plainly: the core pipeline is <strong>deterministic</strong>, and the generative layer
              is optional, opt-in and additive. That ordering is what makes the output defensible.
            </p>
            <Figure
              n="4"
              title="AI pipeline — deterministic core, optional generative layer"
              note="Dashed border marks the only generative component in the default data path."
            >
              <AiDiagram />
            </Figure>

            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <div>
                <h3 className="prd-display mb-2 text-[16px] font-semibold text-stone-900">
                  Design principles
                </h3>
                <dl className="space-y-2.5">
                  {[
                    ['Privacy', 'No audio leaves the deployment boundary by default. Verified.'],
                    ['Determinism first', 'Generative processing is additive, never load-bearing.'],
                    ['Fail soft', 'An LLM failure must degrade to deterministic output, never fail the job.'],
                    ['Preservation', 'The grammar pass may repunctuate; it may not rewrite what was said.'],
                    ['Evaluation', 'No model or prompt change should ship without a measurable comparison.'],
                    ['Lifecycle', 'Model versions pinned and recorded, not drifting with the environment.'],
                  ].map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[110px_1fr] gap-3">
                      <dt className="text-[12px] font-semibold text-stone-800">{k}</dt>
                      <dd className="text-[12.5px] leading-snug text-stone-600">{v}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <div className="prd-note prd-note-warn self-start">
                <p className="mb-2">
                  <strong>The one AI risk that matters most.</strong>
                </p>
                <p className="mb-2">
                  The grammar pass instructs an LLM not to change wording. That constraint is a{' '}
                  <em>prompt</em>, not an enforced check — nothing in the system verifies the model
                  complied. An LLM that quietly substitutes a word changes what a person is recorded
                  as having said.
                </p>
                <p>
                  <strong>AI-001</strong> closes this with an automated word-preservation diff and a
                  revert-on-failure path. Until it exists, the feature should remain opt-in
                  (Decision&nbsp;4).
                </p>
              </div>
            </div>

            <div className="prd-note mt-5">
              <p>
                <strong>No accuracy figure appears anywhere in this document.</strong> None has ever
                been measured for this project — no WER, no DER, no benchmark run. Rather than
                invent one, <strong>AI-004</strong> makes establishing the measurement methodology a
                Must-priority requirement. The first number it produces may well be poor; having it
                is the point.
              </p>
            </div>
          </section>
        )}

        {/* ── 10 · architecture ── */}
        {!execOnly && (
          <section id="architecture" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[9]} />
            <p className="mb-1 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              Shown at product-constraint altitude: what must be true and why it matters. How each
              constraint is implemented belongs in a technical design document or an ADR, not here.
            </p>
            <Figure
              n="5"
              title="System architecture and trust boundary"
              note="Only the frontend origin is externally reachable. The API, GPU layer and optional LLM all stay on loopback — this is what lets the product run on hosts that forward a single port."
            >
              <SystemDiagram />
            </Figure>

            <div className="prd-note mt-4">
              <p>
                <strong>Product / engineering boundary.</strong> This PRD states constraints
                (&ldquo;a stuck job must free its resources&rdquo;). It deliberately does not
                choose the mechanism — subprocess isolation versus a worker pool is a genuine
                architectural decision that belongs to engineering, recorded in an ADR. Where this
                document names an implementation, it is describing what already exists, not
                mandating what must be built.
              </p>
            </div>
          </section>
        )}

        {/* ── 11 · performance ── */}
        {!execOnly && (
          <section id="performance" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[10]} />
            <p className="mb-4 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              The distinction between these three columns is the most important thing on this page.
              Conflating them is how a target becomes quoted as a measurement.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[680px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b-2 border-stone-900">
                    <th className="py-2 pr-4 text-left">
                      <Label>Dimension</Label>
                    </th>
                    <th className="py-2 pr-4 text-left">
                      <Label>Current verified state</Label>
                    </th>
                    <th className="py-2 pr-4 text-left">
                      <Label>Target / proposed</Label>
                    </th>
                    <th className="py-2 text-left">
                      <Label>To be measured</Label>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    [
                      'Upload → job ID',
                      'Immediate — response precedes processing',
                      'Unchanged',
                      '—',
                    ],
                    [
                      'Upload transport',
                      'Expect-header proxy defect fixed (d352874)',
                      'Bounded, documented timeout',
                      'Upload success rate',
                    ],
                    [
                      'Transcription throughput',
                      'Not measured on production GPU',
                      'Stakeholder expectation only — no committed figure',
                      'Real-time factor (PERF-001)',
                    ],
                    [
                      'GPU / VRAM',
                      'Runtime detection, VRAM-sized chunking, cross-engine lock',
                      'Unchanged',
                      'Peak VRAM per job class',
                    ],
                    [
                      'OOM behaviour',
                      'Halve-and-retry + diarization CPU fallback',
                      'Unchanged',
                      'OOM frequency under real load',
                    ],
                    [
                      'Stalled jobs',
                      'Failed at 45 min; worker slot not released',
                      'Automatic worker recovery (REL-002)',
                      'Time-to-recovery',
                    ],
                    [
                      'Grammar pass latency',
                      'Not measured',
                      'Must not dominate total job time',
                      'Added latency per hour of audio',
                    ],
                    [
                      'Disk',
                      'TTL cleanup of outputs and temp files',
                      'Stated retention policy (SEC-005)',
                      'Steady-state disk per active job',
                    ],
                  ].map(([d, c, t, m]) => (
                    <tr key={d} className="border-b border-stone-200 align-top">
                      <td className="py-2 pr-4 font-medium text-stone-800">{d}</td>
                      <td className="py-2 pr-4 text-stone-600">{c}</td>
                      <td className="py-2 pr-4 text-stone-600">{t}</td>
                      <td className="py-2 text-amber-800">{m}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="prd-note prd-note-warn mt-5">
              <p>
                <strong>Explicit non-invention notice.</strong> Project documentation records
                &ldquo;roughly 2&ndash;3 minutes for a 90-minute recording on an RTX 4090-class
                GPU&rdquo; as a <em>target</em>. The engineering guide states plainly that no
                validated benchmark exists. This PRD preserves that distinction rather than
                promoting the figure to a measurement.
              </p>
            </div>
          </section>
        )}

        {/* ── 12 · metrics ── */}
        {!execOnly && (
          <section id="metrics" className="scroll-mt-24 pt-16">
            <SectionHeading s={SECTIONS[11]} />
            <p className="mb-4 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
              Every KPI below is defined by its <em>measurement method</em>. Five of six have no
              baseline, because none has been measured. Establishing them is Phase&nbsp;2 work.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b-2 border-stone-900">
                    {['Group', 'Metric', 'Why it matters', 'How measured', 'Current', 'Target'].map(
                      (h) => (
                        <th key={h} className="py-2 pr-4 text-left last:pr-0">
                          <Label>{h}</Label>
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {KPIS.map((k) => (
                    <tr key={k.metric} className="border-b border-stone-200 align-top">
                      <td className="py-2 pr-4 text-[11px] tracking-wide text-stone-500 uppercase">
                        {k.group}
                      </td>
                      <td className="py-2 pr-4 font-medium text-stone-800">{k.metric}</td>
                      <td className="py-2 pr-4 text-stone-600">{k.why}</td>
                      <td className="py-2 pr-4 text-stone-600">{k.how}</td>
                      <td className="py-2 pr-4 text-stone-500">{k.current}</td>
                      <td className="py-2 font-medium text-amber-800">{k.target}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* ── 13 · roadmap ── */}
        <section id="roadmap" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[12]} />
          <div className="grid gap-4 lg:grid-cols-3">
            {[
              {
                h: 'Shipped',
                s: 'Current state, verified',
                tone: 'teal',
                items: [
                  'Upload, validation, streaming to disk',
                  'ASR — Parakeet on GPU, WhisperX on CPU',
                  'Speaker diarization',
                  'Rule-based meeting intelligence',
                  'Live SSE progress',
                  'Five export formats + bundle',
                  'Same-origin API proxy',
                  'VRAM-aware chunking + OOM recovery',
                ],
              },
              {
                h: 'Next',
                s: 'Closes documented gaps',
                tone: 'amber',
                items: [
                  'FR-006 — durable job store',
                  'PERF-001 — GPU benchmark methodology',
                  'AI-004 — accuracy evaluation methodology',
                  'AI-001 — grammar word-preservation check',
                  'REL-002 — automatic worker recovery',
                  'SEC-005 — stated retention policy',
                ],
              },
              {
                h: 'Future',
                s: 'Decision-gated — not committed',
                tone: 'stone',
                items: [
                  'Multi-user authentication (Decision 1)',
                  'Cloud API /api/v2 completion (Decision 2)',
                  'Speaker identification (no demand evidenced)',
                  'Usage and cost telemetry',
                ],
              },
            ].map((col) => (
              <div key={col.h}>
                <div
                  className={`mb-3 border-t-2 pt-2 ${
                    col.tone === 'teal'
                      ? 'border-teal-800'
                      : col.tone === 'amber'
                        ? 'border-amber-600'
                        : 'border-stone-400'
                  }`}
                >
                  <h3 className="prd-display text-[17px] font-semibold text-stone-900">{col.h}</h3>
                  <p className="text-[11.5px] text-stone-500">{col.s}</p>
                </div>
                <ul className="space-y-1.5">
                  {col.items.map((i) => (
                    <li key={i} className="text-[13px] leading-snug text-stone-700">
                      {i}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <p className="mt-5 text-[12px] text-stone-500">
            &ldquo;Next&rdquo; items were selected because each closes a limitation the project
            already documents about itself — none is new scope invented by this PRD. &ldquo;Future&rdquo;
            items are explicitly not commitments.
          </p>
        </section>

        {/* ── 14 · risks ── */}
        <section id="risks" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[13]} />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[680px] border-collapse text-[12.5px]">
              <thead>
                <tr className="border-b-2 border-stone-900">
                  {['Risk', 'Impact', 'Likelihood', 'Mitigation', 'Owner'].map((h) => (
                    <th key={h} className="py-2 pr-4 text-left last:pr-0">
                      <Label>{h}</Label>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {RISKS.map((r) => (
                  <tr key={r.risk} className="border-b border-stone-200 align-top">
                    <td className="py-2.5 pr-4 font-medium text-stone-800">{r.risk}</td>
                    <td className="py-2.5 pr-4">
                      <span
                        className={`rounded-sm px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase ring-1 ring-inset ${
                          r.impact === 'High'
                            ? 'bg-rose-50 text-rose-900 ring-rose-700/25'
                            : 'bg-stone-100 text-stone-600 ring-stone-400/40'
                        }`}
                      >
                        {r.impact}
                      </span>
                    </td>
                    <td className="py-2.5 pr-4 text-stone-600">{r.likelihood}</td>
                    <td className="py-2.5 pr-4 text-stone-600">{r.mitigation}</td>
                    <td className="py-2.5 text-stone-600">{r.owner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-[12px] text-stone-500">
            Three risks carry <strong>Unknown</strong> likelihood. That is not an omission — it is
            the honest consequence of having no measurement for GPU throughput, grammar-pass
            fidelity, or the intended deployment model. Each resolves once the corresponding
            requirement or decision lands.
          </p>
        </section>

        {/* ── 15 · decisions ── */}
        <section id="decisions" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[14]} />
          <p className="mb-5 max-w-[70ch] text-[14.5px] leading-relaxed text-stone-600">
            These cannot be answered by further analysis of the repository — the answers are not in
            the code. They need a stakeholder.
          </p>

          <ol className="space-y-4">
            {DECISIONS.map((d) => (
              <li key={d.n} className="rounded-sm border border-stone-300 bg-white p-4">
                <div className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="font-mono text-[12px] font-semibold text-teal-800 tabular-nums">
                    D-{String(d.n).padStart(2, '0')}
                  </span>
                  <h3 className="flex-1 text-[15px] leading-snug font-semibold text-stone-900">
                    {d.question}
                  </h3>
                  <span className="rounded-sm bg-rose-50 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-rose-900 uppercase ring-1 ring-rose-700/25 ring-inset">
                    Needs input
                  </span>
                </div>
                <p className="mb-3 text-[13px] leading-relaxed text-stone-600">{d.why}</p>
                <ul className="space-y-1">
                  {d.options.map((o) => (
                    <li key={o} className="flex items-start gap-2 text-[12.5px] text-stone-700">
                      <span aria-hidden="true" className="mt-0.5 text-stone-400">
                        ○
                      </span>
                      {o}
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ol>

          <div className="prd-note mt-5">
            <p>
              <strong>Nothing is recorded as decided.</strong> No decision in this document has been
              ratified by a stakeholder, so none is listed under a &ldquo;Decided&rdquo; heading.
              That section appears once the first decision is made.
            </p>
          </div>
        </section>

        {/* ── 16 · takeaway ── */}
        <section id="takeaway" className="scroll-mt-24 pt-16">
          <SectionHeading s={SECTIONS[15]} />
          <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
            {[
              {
                h: 'What we are building',
                b: 'A self-hosted transcription engine that turns recordings into speaker-labelled transcripts and structured meeting artifacts, with no third party in the data path.',
              },
              {
                h: 'Why it matters',
                b: 'It removes both the per-minute vendor bill and the data-residency conversation from meeting transcription. (Inference — no business case is documented.)',
              },
              {
                h: 'What exists today',
                b: 'Nearly all of it. The full pipeline runs on GPU and CPU, with live progress, five export formats, VRAM-aware scheduling and OOM recovery — deployed and working.',
              },
              {
                h: 'What must happen next',
                b: 'Four things: durable job storage, a GPU benchmark, an accuracy evaluation method, and an enforced fidelity check on the AI grammar pass.',
              },
            ].map((c) => (
              <div key={c.h}>
                <h3 className="prd-display mb-1.5 text-[17px] font-semibold text-stone-900">
                  {c.h}
                </h3>
                <p className="text-[13.5px] leading-relaxed text-stone-600">{c.b}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 border-t-2 border-stone-900 pt-4">
            <Label>The biggest decision required</Label>
            <p className="prd-display mt-2 max-w-[46ch] text-[24px] leading-snug font-semibold text-stone-900">
              Who is this product actually for?
            </p>
            <p className="mt-2 max-w-[68ch] text-[13.5px] leading-relaxed text-stone-600">
              Single operator, small team, or a product for others to use. Everything contested in
              this document — authentication, job persistence priority, the fate of the cloud API,
              half the non-functional requirements — resolves the moment that question is answered,
              and stays open until it is.
            </p>
          </div>

          <footer className="mt-14 border-t border-stone-300 pt-4 text-[11.5px] leading-relaxed text-stone-500">
            Transcript Engine PRD v1.0 · evidence base: repository at commit d352874, plus
            README, PROJECT_ENGINEERING_GUIDE, SECURITY, USER_GUIDE and DEPLOYMENT. Labels
            distinguish verified implementation from inference, planned work and unverified claims
            throughout. No metric, benchmark, user count, accuracy figure or business outcome in
            this document was invented.
          </footer>
        </section>
      </main>
    </div>
  );
}

/**
 * Scoped styles.
 *
 * Kept here rather than in globals.css so the PRD route stays self-contained —
 * it must not alter the transcription app's styling. The display face is a
 * system serif stack rather than a webfont: the app is routinely built on
 * offline or network-restricted GPU hosts, where a font fetch at build time is
 * a silent failure waiting to happen.
 */
const PRD_CSS = `
.prd-root .prd-display{
  font-family: Charter, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  letter-spacing: -0.011em;
}
.prd-root .prd-note{
  border-left: 2px solid #0f766e;
  background: #f4f6f5;
  padding: 0.85rem 1rem;
  font-size: 13px;
  line-height: 1.65;
  color: #44403c;
  max-width: 78ch;
}
.prd-root .prd-note-warn{
  border-left-color: #b45309;
  background: #fbf7f0;
}
.prd-root .prd-navscroll{ scrollbar-width: none; }
.prd-root .prd-navscroll::-webkit-scrollbar{ display: none; }
.prd-root .prd-navmask{
  -webkit-mask-image: linear-gradient(to right, #000 0, #000 calc(100% - 28px), transparent 100%);
  mask-image: linear-gradient(to right, #000 0, #000 calc(100% - 28px), transparent 100%);
}
.prd-root .prd-caret{ display: inline-block; }

@media (prefers-reduced-motion: reduce){
  .prd-root *{ scroll-behavior: auto !important; }
}

@media print{
  .prd-root .prd-nav{ display: none !important; }
  .prd-root .prd-controls{ display: none !important; }
  .prd-root .prd-caret{ display: none !important; }
  /* Expand every collapsed requirement so the PDF is complete. */
  .prd-root .prd-detail{ display: block !important; }
  .prd-root section{ break-inside: avoid-page; padding-top: 1.5rem !important; }
  .prd-root h1, .prd-root h2, .prd-root h3{ break-after: avoid-page; }
  .prd-root table{ font-size: 10px; }
  .prd-root .overflow-x-auto{ overflow: visible !important; }
  .prd-root [class*="min-w-["]{ min-width: 0 !important; }
  .prd-root{ background: #fff !important; }
}
`;
