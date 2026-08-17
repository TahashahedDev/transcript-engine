export interface Job {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  profile: string;
  mode: string;
  audio_filename: string;
  created_at: string;
  completed_at: string | null;
  error: string | null;
  artifacts: string[] | null;
  metrics: JobMetrics | null;
}

export interface JobMetrics {
  audio_duration_seconds: number;
  processing_time_seconds: number;
  word_count: number;
  speaker_count: number;
  corrections_applied: number;
  action_items: number;
  decisions: number;
  questions_total: number;
  open_questions: number;
  named_entities: number;
  language: string;
  [key: string]: unknown;
}

export interface ProgressEvent {
  type: 'stage' | 'completed' | 'error' | 'audio_info' | 'resource' | 'pipeline_decision' | 'partial_transcript' | 'chunk_progress';
  stage?: string;
  fraction?: number;
  message?: string;
  audio_duration?: number;
  cpu_pct?: number;
  ram_gb?: number;
  // pipeline_decision
  decisions?: string[];
  savings_s?: number;
  // partial_transcript
  segments?: Array<{ speaker: string; text: string; start: number }>;
  total_segments?: number;
  word_count?: number;
  // chunk_progress (Parakeet chunked inference)
  current_chunk?: number;
  total_chunks?: number;
}

export interface ProcessingReport {
  job_id: string;
  audio_file: string;
  audio_duration_s: number;
  started_at: string;
  completed_at: string;
  total_s: number;
  status: string;
  upload_s: number;
  stages: Record<string, { start_s: number; end_s: number; duration_s: number; pct: number }>;
  models: { asr: string; backend: string; device: string; diarization: string };
  output_validation: { total: number; present: number; missing: string[]; artifacts?: Record<string, { status: string; size_bytes?: number }> };
  temp_validation: { audio_deleted: boolean; temp_dir_deleted: boolean; orphaned_chunks?: number };
  error: string | null;
}

export interface ResourceUsage {
  job_id: string;
  interval_s: number;
  samples: Array<{ t_s: number; stage: string; cpu_pct: number; ram_gb: number; disk_used_gb: number; threads: number }>;
  summary: { cpu_peak: number; cpu_avg: number; ram_peak_gb: number; ram_avg_gb: number };
}

export interface DiagnosticsResponse {
  job_id: string;
  status: string;
  processing_report: ProcessingReport | null;
  resource_usage: ResourceUsage | null;
}

export interface HFTokenStatus {
  present: boolean;
  valid: boolean | null;
  diarization_access: boolean | null;
  segmentation_access: boolean | null;
  error: string | null;
}

export interface ModelInfo {
  name: string;
  cached: boolean;
  size_mb: number | null;
}

export interface GpuStatus {
  available: boolean;
  name: string | null;
  cuda_version: string | null;
  vram_total_gb: number | null;
  vram_free_gb: number | null;
  compute_capability: string | null;
  compatibility_error: string | null;
}

export interface SetupCheckResponse {
  python_version: string;
  ffmpeg: boolean;
  ffmpeg_version: string | null;
  hf_token: HFTokenStatus;
  profiles: string[];
  output_dir_writable: boolean;
  whisper_models: ModelInfo[];
  alignment_models: ModelInfo[];
  ready: boolean;
  asr_backend: string;
  parakeet_model: string | null;
  max_upload_mb: number;
  gpu: GpuStatus;
  ai_grammar_available: boolean;
}

export interface ArtifactsResponse {
  files: string[];
  bundle: string | null;
}

export interface SearchIndexSegment {
  start: number;
  end: number;
  speaker_id: string;
  speaker_display: string;
  text: string;
  words?: Array<{ text: string; start: number; end: number; confidence: number }>;
}

export interface SearchIndex {
  version: number;
  audio_file: string;
  duration: number;
  language: string;
  profile: string;
  speakers: Record<string, string>;
  segments: SearchIndexSegment[];
  topics?: Array<{ label: string; start: number; end: number }>;
  action_items?: Array<{ text: string; speaker_id: string; timestamp: number }>;
  decisions?: Array<{ text: string; speaker_id: string; timestamp: number }>;
  questions?: Array<{ text: string; speaker_id: string; timestamp: number }>;
}
