/**
 * PRD content model.
 *
 * Separated from the page so the document's *facts* can be reviewed and edited
 * without reading through layout code. Every requirement carries an explicit
 * evidence status — the PRD's central discipline is that a reader can always
 * tell an implemented capability from a proposal.
 *
 * Source: synthesis of the PRD methodology document and the Transcript Engine
 * PRD, both grounded in the repository at commit d352874.
 */

export type Priority = 'must' | 'should' | 'could' | 'wont';

/** Evidence status. Deliberately not a boolean — "partial" and "gated" are the
 *  two states that a naive implemented/not-implemented split would misreport. */
export type Status =
  | 'verified'
  | 'partial'
  | 'gap'
  | 'planned'
  | 'gated'
  | 'unverified';

export type Category =
  | 'Functional'
  | 'UX'
  | 'AI/ML'
  | 'Platform'
  | 'Performance'
  | 'Security'
  | 'Reliability'
  | 'Architecture';

export interface Requirement {
  id: string;
  category: Category;
  requirement: string;
  rationale: string;
  priority: Priority;
  acceptance: string;
  status: Status;
  evidence: string;
}

export const STATUS_LABEL: Record<Status, string> = {
  verified: 'Verified',
  partial: 'Partial',
  gap: 'Gap',
  planned: 'Planned',
  gated: 'Decision-gated',
  unverified: 'Not verified',
};

export const PRIORITY_LABEL: Record<Priority, string> = {
  must: 'Must',
  should: 'Should',
  could: 'Could',
  wont: "Won't (this phase)",
};

export const REQUIREMENTS: Requirement[] = [
  // ── Functional ────────────────────────────────────────────────────────────
  {
    id: 'FR-001',
    category: 'Functional',
    requirement:
      'Accept audio/video upload in supported formats and return a job ID without waiting for transcription to complete.',
    rationale:
      'A user must never watch a blocked request while a GPU job runs; job acceptance and job processing are separate concerns.',
    priority: 'must',
    acceptance:
      'POST returns job_id + status:queued within the file transfer time. No processing-time wait.',
    status: 'verified',
    evidence: 'api/routes/jobs.py',
  },
  {
    id: 'FR-002',
    category: 'Functional',
    requirement: 'Report live, stage-level progress during processing.',
    rationale:
      'Transcription can run for many minutes. A generic spinner gives no signal whether work is progressing or wedged.',
    priority: 'must',
    acceptance: 'UI reflects the current pipeline stage within 5s of a transition, over SSE.',
    status: 'verified',
    evidence: 'web/hooks/useProgress.ts, api/progress_store.py',
  },
  {
    id: 'FR-003',
    category: 'Functional',
    requirement: 'Produce a speaker-labelled transcript with word-level timestamps.',
    rationale:
      'Word-level timing is what makes subtitles, search, and per-speaker attribution possible downstream.',
    priority: 'must',
    acceptance: "Every completed job's transcript.json includes per-word start/end and a speaker ID.",
    status: 'verified',
    evidence: 'transcript_engine/models/transcript.py',
  },
  {
    id: 'FR-004',
    category: 'Functional',
    requirement:
      'Generate summary, action items, decisions, open questions and topic timeline from the transcript.',
    rationale:
      'The product promise is meeting *outcomes*, not raw text — most readers never read the full transcript.',
    priority: 'must',
    acceptance: 'All five artifact files present for a completed job.',
    status: 'verified',
    evidence: 'transcript_engine/intelligence/engine.py',
  },
  {
    id: 'FR-005',
    category: 'Functional',
    requirement: 'Allow individual and bundled (ZIP) artifact download.',
    rationale: 'Different consumers need different formats; a bundle avoids ten separate downloads.',
    priority: 'must',
    acceptance: 'Each artifact downloadable standalone and via transcript_bundle.zip.',
    status: 'verified',
    evidence: 'README "Output"',
  },
  {
    id: 'FR-006',
    category: 'Functional',
    requirement: 'Persist job records durably across a server restart.',
    rationale:
      'Jobs are currently in memory only — any restart or crash silently destroys user work and history.',
    priority: 'must',
    acceptance:
      'A job created before restart is queryable by ID after restart, with correct terminal state.',
    status: 'gap',
    evidence: 'PostgreSQL schema exists in db/, not wired to the live job store',
  },
  {
    id: 'FR-007',
    category: 'Functional',
    requirement:
      'Reject unsupported file types and over-limit uploads with a clear, actionable error before processing starts.',
    rationale: 'Failing late wastes GPU time and gives the user a worse error further from the cause.',
    priority: 'must',
    acceptance: '400/413 with a specific reason; file never written to permanent storage.',
    status: 'verified',
    evidence: 'api/routes/jobs.py',
  },
  {
    id: 'FR-008',
    category: 'Functional',
    requirement:
      'Support multiple quality/processing modes trading speed for post-processing depth.',
    rationale: 'A quick draft and an archive-quality transcript are genuinely different jobs.',
    priority: 'should',
    acceptance: 'Selecting a mode changes which processors run and is reflected in the output.',
    status: 'verified',
    evidence: 'PIPELINE_MODES in transcript_engine/config/settings.py',
  },
  {
    id: 'FR-009',
    category: 'Functional',
    requirement: 'Apply optional domain vocabulary correction profiles.',
    rationale:
      'General ASR reliably mis-hears domain terminology; a banking profile already exists as proof of the need.',
    priority: 'should',
    acceptance:
      'Selecting a profile measurably changes terminology correction behaviour on domain audio.',
    status: 'verified',
    evidence: 'vocabulary processor + banking profile',
  },
  {
    id: 'FR-010',
    category: 'Functional',
    requirement: 'Allow deletion of a job and its artifacts.',
    rationale: 'Recordings are sensitive; the operator must be able to remove them on demand.',
    priority: 'should',
    acceptance: 'DELETE /api/jobs/{id} removes output/temp directories and the job record.',
    status: 'verified',
    evidence: 'api/routes/jobs.py',
  },

  // ── UX ────────────────────────────────────────────────────────────────────
  {
    id: 'UX-001',
    category: 'UX',
    requirement:
      'The upload control must state accepted formats and the real server-enforced size limit.',
    rationale:
      'A hardcoded client limit drifts from the server and rejects files locally that would upload fine — this actually happened.',
    priority: 'must',
    acceptance: 'Limit shown matches the /api/setup/check response, not a client constant.',
    status: 'verified',
    evidence: 'web/components/UploadZone.tsx',
  },
  {
    id: 'UX-002',
    category: 'UX',
    requirement:
      "When a mode's differentiating feature is unavailable, say so rather than silently producing identical output.",
    rationale:
      'Offering a slower mode whose result is byte-identical to the cheaper one is a trust failure, not a minor cosmetic issue.',
    priority: 'must',
    acceptance: 'UI states "produces the same result as [cheaper mode]" when the feature is off.',
    status: 'verified',
    evidence: 'web/app/page.tsx',
  },
  {
    id: 'UX-003',
    category: 'UX',
    requirement: 'Setup/readiness state must be visible before a user attempts an upload that will fail.',
    rationale: 'Missing ffmpeg or an unaccepted model licence should surface at setup, not mid-job.',
    priority: 'must',
    acceptance: 'Missing ffmpeg / HF token / output dir surfaced on the setup page.',
    status: 'verified',
    evidence: 'api/routes/setup.py — /setup/check',
  },
  {
    id: 'UX-004',
    category: 'UX',
    requirement:
      'Progress UI must reflect genuine backend stage and must not imply progress on a dead connection.',
    rationale:
      'A bar that keeps creeping on a broken stream reports progress the system cannot actually observe.',
    priority: 'should',
    acceptance:
      'SSE reconnect caps retries and surfaces "connection lost" as distinct from job failure.',
    status: 'verified',
    evidence: 'web/hooks/useProgress.ts',
  },

  // ── AI/ML ─────────────────────────────────────────────────────────────────
  {
    id: 'AI-001',
    category: 'AI/ML',
    requirement:
      'The grammar-correction pass must not alter transcript wording — only punctuation, capitalisation and sentence boundaries.',
    rationale:
      'This is the single highest-integrity risk in the product: an LLM editing a transcript can change what a person is recorded as having said. Today the constraint is a prompt instruction, not an enforced check.',
    priority: 'must',
    acceptance:
      'Automated diff: word sequence (case/punctuation-insensitive) identical before/after on a test corpus; failures logged and the segment reverted to the pre-correction text.',
    status: 'gap',
    evidence: 'transcript_engine/processors/ai_grammar.py — prompt-only constraint',
  },
  {
    id: 'AI-002',
    category: 'AI/ML',
    requirement:
      'A failure in the optional AI pass must degrade to deterministic output, never fail the job.',
    rationale: 'An optional enhancement must never take down the core deliverable.',
    priority: 'must',
    acceptance:
      'Simulated LLM-endpoint failure still yields a completed job with an unmodified transcript.',
    status: 'partial',
    evidence: 'Per-segment try/except exists in ai_grammar.py; not covered by an explicit test',
  },
  {
    id: 'AI-003',
    category: 'AI/ML',
    requirement: 'Indicate clearly to the user when a generative pass was applied.',
    rationale:
      'A reader must be able to tell a machine-edited transcript from a raw one when judging its fidelity.',
    priority: 'should',
    acceptance:
      'The completed-job view states whether the grammar pass ran, matching the existing pre-upload signal.',
    status: 'partial',
    evidence: 'Pre-upload state shown in web/app/page.tsx; not surfaced post-job',
  },
  {
    id: 'AI-004',
    category: 'AI/ML',
    requirement:
      'Establish a transcription accuracy (WER) and diarization accuracy (DER) evaluation methodology.',
    rationale:
      'No accuracy number exists anywhere in the project. Without a measurement method, no model or prompt change can be assessed as an improvement or a regression.',
    priority: 'must',
    acceptance:
      'A documented, repeatable process over a labelled reference set that produces a number. Establishing the metric is the requirement — not achieving a specific score.',
    status: 'gap',
    evidence: 'No evaluation harness present in the repository',
  },
  {
    id: 'AI-005',
    category: 'AI/ML',
    requirement:
      'Pin and record model versions for ASR, diarization and the grammar LLM.',
    rationale: 'A silent model bump can change every output; it should be a reviewable change.',
    priority: 'should',
    acceptance: 'A model swap is a recorded, reviewable change rather than environment drift.',
    status: 'gap',
    evidence: 'Model IDs are configurable via env; no change-record process exists',
  },

  // ── Platform ──────────────────────────────────────────────────────────────
  // NFR-003 and NFR-004 from the source PRD are deliberately absent: they
  // duplicated REL-003 (OOM survival) and REL-002 (hung-job recovery) exactly,
  // and are merged there rather than restated under a second ID.
  {
    id: 'NFR-001',
    category: 'Platform',
    requirement:
      'Run correctly on any CUDA GPU by detecting available VRAM at runtime, with no hardcoded hardware assumptions.',
    rationale:
      'The product is deployed on rented GPUs whose model is not known in advance. Hardcoding a card would break every deployment that differs from the developer machine.',
    priority: 'must',
    acceptance:
      'Chunk sizing derives from runtime GPU detection; no GPU model name appears anywhere in source.',
    status: 'verified',
    evidence: 'transcript_engine/gpu/hardware.py',
  },
  {
    id: 'NFR-002',
    category: 'Platform',
    requirement: 'Run fully on CPU when no GPU is present.',
    rationale:
      'Keeps the product usable for evaluation and light workloads without any GPU spend.',
    priority: 'must',
    acceptance: 'The Whisper/WhisperX backend is selected automatically when no CUDA device exists.',
    status: 'verified',
    evidence: 'transcript_engine/config/settings.py, README',
  },
  {
    id: 'NFR-005',
    category: 'Platform',
    requirement:
      'Concurrent GPU-resident stages must not exceed available VRAM.',
    rationale:
      'Transcription and diarization are each sized to fit alone; run together on a small card they collide. This was a real production OOM source, not a theoretical one.',
    priority: 'must',
    acceptance: 'No OOM when both engines would be CUDA-resident simultaneously.',
    status: 'verified',
    evidence: 'GPU_COMPUTE_LOCK in transcript_engine/gpu/hardware.py',
  },
  {
    id: 'NFR-006',
    category: 'Platform',
    requirement:
      'The web UI must work when reached at any host address, not only localhost.',
    rationale:
      'A build made on a remote GPU host must not ship a frontend that points at the visitor\'s own machine — this previously broke every remote deployment.',
    priority: 'must',
    acceptance: 'Same-origin API resolution works from whatever origin the page was served on.',
    status: 'verified',
    evidence: 'web/lib/apiBase.ts + web/app/api/[...path]/route.ts',
  },

  // ── Performance ───────────────────────────────────────────────────────────
  {
    id: 'PERF-001',
    category: 'Performance',
    requirement:
      'Establish a repeatable GPU transcription benchmark (real-time factor) on a named reference GPU.',
    rationale:
      "Every performance claim the project might make depends on this existing first. The project's own docs name it as the next required step.",
    priority: 'must',
    acceptance:
      'Documented RTF for a fixed reference audio file on a named GPU, reproducible run-to-run within a stated tolerance.',
    status: 'gap',
    evidence: 'PROJECT_ENGINEERING_GUIDE.md §13 — target documented, no measurement captured',
  },
  {
    id: 'PERF-002',
    category: 'Performance',
    requirement:
      'Large-file uploads must complete or fail within a bounded, documented time — never hang silently.',
    rationale:
      'A 30-minute silent "Uploading…" with no terminal state is indistinguishable from a hung product.',
    priority: 'must',
    acceptance:
      'No code path leaves an upload pending without client feedback beyond a documented timeout.',
    status: 'partial',
    evidence:
      'Proxy Expect-header defect fixed at commit d352874; Node default requestTimeout (300s) still governs, undocumented',
  },
  {
    id: 'PERF-003',
    category: 'Performance',
    requirement:
      'A 60-minute recording should transcribe well under real-time on GPU hardware.',
    rationale:
      'Stakeholder expectation, recorded so it is not lost — but it is not yet a committed target.',
    priority: 'should',
    acceptance:
      'No numeric target is set in this document. To be established only once PERF-001 provides a baseline.',
    status: 'unverified',
    evidence: 'Stakeholder expectation requiring validation — not a measured or committed figure',
  },

  // ── Security ──────────────────────────────────────────────────────────────
  {
    id: 'SEC-001',
    category: 'Security',
    requirement: 'Audio and derived artifacts must not be transmitted to any third party by default.',
    rationale: 'This is the product\'s core differentiator, not merely a policy preference.',
    priority: 'must',
    acceptance:
      'ASR and diarization run locally; the grammar pass calls only an operator-configured endpoint, defaulting to local Ollama.',
    status: 'verified',
    evidence: 'transcript_engine/processors/ai_grammar.py, .env.example',
  },
  {
    id: 'SEC-002',
    category: 'Security',
    requirement: 'Secrets must never be logged, echoed, or returned via API responses.',
    rationale: 'A leaked HF token or API key in a log is an immediate compromise.',
    priority: 'must',
    acceptance: 'Token endpoints never return or log the value.',
    status: 'verified',
    evidence: 'api/routes/setup.py — documented and implemented',
  },
  {
    id: 'SEC-003',
    category: 'Security',
    requirement: 'Vulnerabilities must have a private disclosure path.',
    rationale: 'Public-issue disclosure of a live vulnerability harms users before a fix exists.',
    priority: 'must',
    acceptance: 'A documented private advisory process exists.',
    status: 'verified',
    evidence: 'SECURITY.md',
  },
  {
    id: 'SEC-004',
    category: 'Security',
    requirement: 'Restrict API access to authorised users when deployed beyond a single operator.',
    rationale:
      'Acceptable to omit for a single-operator tool; a serious exposure the moment the deployment is shared.',
    priority: 'could',
    acceptance: 'Gated on Decision 1 — cannot be specified before the user model is confirmed.',
    status: 'gated',
    evidence: 'No auth on the primary API today',
  },
  {
    id: 'SEC-005',
    category: 'Security',
    requirement: 'Uploaded audio and transcripts must have a defined retention/deletion policy.',
    rationale: 'Recordings are among the most sensitive data a business holds; "until the disk fills" is not a policy.',
    priority: 'should',
    acceptance: 'A stated retention period, enforced automatically, documented for the operator.',
    status: 'partial',
    evidence: 'TE_API_OUTPUT_TTL_HOURS exists; no policy statement',
  },

  // ── Reliability ───────────────────────────────────────────────────────────
  {
    id: 'REL-001',
    category: 'Reliability',
    requirement: 'A stalled job must reach a terminal state automatically.',
    rationale: 'Otherwise the UI waits forever on work that will never finish.',
    priority: 'must',
    acceptance: 'A job with no progress for the configured window is marked failed.',
    status: 'verified',
    evidence: 'Stall watchdog, TE_API_STALL_TIMEOUT_MINUTES (default 45)',
  },
  {
    id: 'REL-002',
    category: 'Reliability',
    requirement: 'The worker pool must recover after a hung job without operator intervention.',
    rationale:
      'Today one wedged job blocks every subsequent job until someone restarts the process — a single failure degrades the whole service.',
    priority: 'should',
    acceptance: 'After a stall is detected, the next queued job starts without a process restart.',
    status: 'gap',
    evidence: 'PROJECT_ENGINEERING_GUIDE.md §14 — Python cannot kill the worker thread',
  },
  {
    id: 'REL-003',
    category: 'Reliability',
    requirement: 'GPU out-of-memory on a single stage must not fail the entire job.',
    rationale: 'OOM is an expected condition on constrained GPUs, not an exceptional one.',
    priority: 'must',
    acceptance: 'Halve-and-retry per chunk, plus one-shot diarization CPU fallback.',
    status: 'verified',
    evidence: 'PROJECT_ENGINEERING_GUIDE.md §14',
  },
  {
    id: 'REL-004',
    category: 'Reliability',
    requirement: 'Operators must see real-time resource state for a running job.',
    rationale: 'Diagnosing a slow job without VRAM/CPU visibility is guesswork.',
    priority: 'should',
    acceptance: 'CPU, RAM and VRAM telemetry available for an in-flight job.',
    status: 'partial',
    evidence: 'cpu_pct / ram_gb over SSE verified; VRAM telemetry not re-verified in this audit',
  },

  // ── Architecture ──────────────────────────────────────────────────────────
  {
    id: 'ARCH-001',
    category: 'Architecture',
    requirement:
      'The backend API port must never require direct external exposure.',
    rationale:
      'On rented/shared GPU hosts, every externally reachable port is attack surface. Only the frontend origin should be reachable.',
    priority: 'must',
    acceptance: 'The browser reaches the API only through the frontend origin.',
    status: 'verified',
    evidence: 'web/app/api/[...path]/route.ts — same-origin proxy',
  },
  {
    id: 'ARCH-002',
    category: 'Architecture',
    requirement: 'Long-running work must not block the HTTP request/response cycle.',
    rationale: 'Ties directly to FR-001 — job acceptance must be independent of job duration.',
    priority: 'must',
    acceptance: 'The response is sent before pipeline execution begins.',
    status: 'verified',
    evidence: 'api/routes/jobs.py — asyncio.create_task',
  },
  {
    id: 'ARCH-003',
    category: 'Architecture',
    requirement: 'Job state must survive a process restart.',
    rationale: 'Product-level constraint behind FR-006; the persistence mechanism is an engineering choice.',
    priority: 'must',
    acceptance: 'State survives restart. Which store, and the migration path, belong in a design doc.',
    status: 'gap',
    evidence: 'Schema exists; wiring plan requires an ADR',
  },
  {
    id: 'ARCH-004',
    category: 'Architecture',
    requirement: 'A stuck job must eventually free system resources for other jobs.',
    rationale:
      'States the constraint behind REL-002 without prescribing the solution — subprocess isolation vs. worker pool is an engineering decision, not a product one.',
    priority: 'should',
    acceptance: 'Resources are reclaimed automatically. Mechanism deliberately unspecified here.',
    status: 'gap',
    evidence: 'Open ADR',
  },
];

// ── KPI framework ───────────────────────────────────────────────────────────

export interface Kpi {
  group: string;
  metric: string;
  why: string;
  how: string;
  current: string;
  target: string;
}

export const KPIS: Kpi[] = [
  {
    group: 'Quality',
    metric: 'Transcription accuracy (WER)',
    why: 'The core product promise. Every downstream artifact inherits transcription errors.',
    how: 'AI-004 methodology over a labelled reference set, run per model change',
    current: 'None — not measured',
    target: 'TBD after baseline',
  },
  {
    group: 'Quality',
    metric: 'Diarization accuracy (DER)',
    why: 'Speaker attribution errors are more damaging than word errors — they misquote people.',
    how: 'AI-004 methodology over labelled reference audio',
    current: 'None — not measured',
    target: 'TBD after baseline',
  },
  {
    group: 'Performance',
    metric: 'Real-time factor',
    why: 'Determines whether the product is usable for long recordings and how much GPU time costs.',
    how: 'PERF-001 benchmark harness on a named reference GPU',
    current: 'None — not measured',
    target: 'TBD after baseline',
  },
  {
    group: 'Reliability',
    metric: 'Job completion rate',
    why: 'Measures whether jobs reach a terminal state without an operator having to intervene.',
    how: 'Log-derived from the job manager over a defined window',
    current: 'Not instrumented',
    target: 'TBD',
  },
  {
    group: 'Product',
    metric: 'Upload success rate',
    why: 'The first step of every session; a failure here means the product never starts.',
    how: 'Proxy and API log analysis',
    current: 'Not instrumented (a defect in this path was fixed at d352874)',
    target: 'TBD',
  },
  {
    group: 'AI',
    metric: 'Grammar-pass fidelity',
    why: 'Detects the highest-integrity risk: an LLM silently changing transcript wording.',
    how: 'AI-001 automated word-preservation diff over a test corpus',
    current: 'Not measured — prompt-only constraint',
    target: '100% word preservation (definitionally, not aspirationally)',
  },
];

// ── Risks ───────────────────────────────────────────────────────────────────

export interface Risk {
  risk: string;
  impact: 'High' | 'Medium' | 'Low';
  likelihood: 'High' | 'Medium' | 'Low' | 'Unknown';
  mitigation: string;
  owner: string;
}

export const RISKS: Risk[] = [
  {
    risk: 'Server restart during active use destroys all job history',
    impact: 'High',
    likelihood: 'High',
    mitigation: 'FR-006 — wire the existing PostgreSQL schema to the live job store',
    owner: 'Engineering',
  },
  {
    risk: 'Generative grammar pass silently alters transcript meaning',
    impact: 'High',
    likelihood: 'Unknown',
    mitigation: 'AI-001 — automated word-preservation check with revert on failure',
    owner: 'Engineering + Decision 4',
  },
  {
    risk: 'Production GPU throughput materially worse than the documented target',
    impact: 'High',
    likelihood: 'Unknown',
    mitigation: 'PERF-001 — establish the benchmark before making any performance claim',
    owner: 'Engineering + Decision 3',
  },
  {
    risk: 'A single hung job degrades service for all subsequent jobs',
    impact: 'Medium',
    likelihood: 'Medium',
    mitigation: 'REL-002 / ARCH-004 — automatic worker recovery',
    owner: 'Engineering',
  },
  {
    risk: 'Unauthenticated API exposed beyond intended single-operator use',
    impact: 'High',
    likelihood: 'Unknown',
    mitigation: 'SEC-004 — blocked until the target user model is confirmed',
    owner: 'Stakeholder — Decision 1',
  },
  {
    risk: 'Further reverse-proxy header-forwarding defects',
    impact: 'Medium',
    likelihood: 'Medium',
    mitigation: 'Add a regression test for the proxy path; one instance already found and fixed',
    owner: 'Engineering',
  },
];

// ── Decisions ───────────────────────────────────────────────────────────────

export interface Decision {
  n: number;
  state: 'open' | 'decided' | 'blocked';
  question: string;
  why: string;
  options: string[];
}

export const DECISIONS: Decision[] = [
  {
    n: 1,
    state: 'open',
    question: 'Who is the target user — single operator, small team, or multi-tenant product?',
    why: 'Gates SEC-004 (auth), the priority of FR-006, and effectively half the non-functional requirement set. Nothing in the repository answers this.',
    options: [
      'Single operator — current architecture is already sufficient',
      'Small trusted team — needs FR-006 plus basic access control',
      'Multi-tenant product — a substantially larger security workstream; effectively a different PRD',
    ],
  },
  {
    n: 2,
    state: 'open',
    question: 'Is the /api/v2 cloud API an active roadmap item or dormant scaffolding?',
    why: 'Endpoints, schema and storage wiring exist, but no worker consumes the queue — a job created there stays queued forever. Its status changes Phase 2 scope.',
    options: ['Finish and staff it', 'Explicitly deprioritise and document as dormant', 'Remove it if abandoned'],
  },
  {
    n: 3,
    state: 'open',
    question: 'Which GPU class should the first performance benchmark target?',
    why: 'PERF-001 is not actionable without a named reference device.',
    options: ['The RTX 3080 currently deployed (default absent other instruction)', 'A different reference card'],
  },
  {
    n: 4,
    state: 'open',
    question: 'Should the AI grammar pass ever be on by default?',
    why: 'An opt-in, disclosed generative feature is materially lower risk than a default-on one. Changes the urgency of AI-001.',
    options: [
      'Keep strictly opt-in (current behaviour)',
      "Default-on only after AI-001's safety check exists",
    ],
  },
  {
    n: 5,
    state: 'open',
    question: 'Is there a business driver behind this PRD, or is production-hardening itself the goal?',
    why: 'Success metrics and any future roadmap cannot be grounded without it. No business-case evidence exists in the repository.',
    options: ['A stated business/ROI driver', 'Technical and operational maturity is the goal'],
  },
];
