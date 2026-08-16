'use client';

import { useEffect, useRef, useState } from 'react';
import { resolveApiBase } from '@/lib/apiBase';
import type { ProgressEvent } from '@/lib/types';

// Must match the stage labels api/pipeline_runner.py's _STAGE_LABELS actually
// emits over SSE (backend truth) — both "Loading Whisper model" and
// "Loading Parakeet model" progress messages are normalized to "Loading
// Model" server-side, regardless of which ASR backend is active.
export const STAGES = [
  'Uploading',
  'Preprocessing Audio',
  'Loading Model',
  'Transcribing',
  'Aligning Words',
  'Loading Diarization',
  'Speaker Diarization',
  'Processing',
  'Vocabulary Corrections',
  'Meeting Intelligence',
  'Generating Files',
  'Creating Bundle',
  'Completed',
];

// Generous expected durations per stage (seconds).  Only used for the
// "taking longer than expected" warning — never for time estimates.
export const STAGE_EXPECTED_SECONDS: Record<string, number> = {
  'Uploading': 30,
  'Preprocessing Audio': 30,
  'Loading Model': 45,
  'Transcribing': 360,
  'Aligning Words': 90,
  'Loading Diarization': 45,
  'Speaker Diarization': 180,
  'Processing': 60,
  'Vocabulary Corrections': 30,
  'Meeting Intelligence': 60,
  'Generating Files': 20,
  'Creating Bundle': 10,
};

// SSE reconnection. A dropped connection (server restart, proxy timeout,
// network blip) must not leave the UI stuck showing progress forever, so we
// retry with backoff instead of closing permanently. After the final attempt
// we stop claiming progress and let useJob's polling decide the job's real
// terminal state — the backend remains the source of truth either way.
const SSE_MAX_RETRIES = 5;
const SSE_RETRY_BASE_MS = 2_000;

export function useProgress(jobId: string | null) {
  const [currentStage, setCurrentStage] = useState('Uploading');
  const [fraction, setFraction] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [cpuPct, setCpuPct] = useState<number | null>(null);
  const [ramGb, setRamGb] = useState<number | null>(null);
  const [stageStartedAt, setStageStartedAt] = useState<number>(() => Date.now());
  const [pipelineDecisions, setPipelineDecisions] = useState<string[]>([]);
  const [partialTranscript, setPartialTranscript] = useState<Array<{ speaker: string; text: string; start: number }> | null>(null);
  const [connectionLost, setConnectionLost] = useState(false);
  const stageRef = useRef<string>('Uploading');
  const lastUpdateRef = useRef<number>(0);
  const liveRef = useRef<boolean>(false);

  useEffect(() => {
    if (!jobId) return;

    lastUpdateRef.current = Date.now();
    const apiUrl = resolveApiBase();

    let es: EventSource | null = null;
    let retries = 0;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    let cancelled = false;

    function connect() {
      if (cancelled) return;

      const source = new EventSource(`${apiUrl}/api/jobs/${jobId}/progress`);
      es = source;

      source.onopen = () => {
        retries = 0;
        liveRef.current = true;
        setConnectionLost(false);
      };

      source.onmessage = (e) => {
        lastUpdateRef.current = Date.now();
        try {
          const event: ProgressEvent = JSON.parse(e.data);

          if (event.type === 'audio_info') {
            if (event.audio_duration != null) setAudioDuration(event.audio_duration);
          } else if (event.type === 'stage') {
            if (event.stage) {
              if (event.stage !== stageRef.current) {
                stageRef.current = event.stage;
                setStageStartedAt(Date.now());
              }
              setCurrentStage(event.stage);
            }
            if (event.fraction !== undefined) setFraction(event.fraction);
          } else if (event.type === 'resource') {
            if (event.cpu_pct != null) setCpuPct(event.cpu_pct);
            if (event.ram_gb != null) setRamGb(event.ram_gb);
          } else if (event.type === 'pipeline_decision') {
            if (event.decisions) setPipelineDecisions(event.decisions);
          } else if (event.type === 'partial_transcript') {
            if (event.segments) setPartialTranscript(event.segments);
          } else if (event.type === 'completed') {
            // Terminal: stop reconnecting.
            cancelled = true;
            liveRef.current = false;
            setCompleted(true);
            setFraction(1);
            setCurrentStage('Completed');
            source.close();
          } else if (event.type === 'error') {
            cancelled = true;
            liveRef.current = false;
            setError(event.message ?? 'Unknown error');
            source.close();
          }
        } catch {
          // Malformed event — skip it rather than tearing down the stream.
        }
      };

      source.onerror = () => {
        source.close();
        liveRef.current = false;
        if (cancelled) return;

        if (retries >= SSE_MAX_RETRIES) {
          // Give up on live progress. Deliberately does NOT set `error`: the
          // job itself may still be running fine server-side, and useJob's
          // polling reports the real outcome. Surfacing a fake failure here
          // would be worse than showing a degraded-connection notice.
          setConnectionLost(true);
          return;
        }
        retries += 1;
        retryTimer = setTimeout(connect, SSE_RETRY_BASE_MS * retries);
      };
    }

    connect();

    return () => {
      cancelled = true;
      liveRef.current = false;
      clearTimeout(retryTimer);
      es?.close();
    };
  }, [jobId]);

  // Heartbeat: if the progress bar has been frozen for >10s during a known-slow
  // stage, nudge the fraction forward so the user knows work is happening.
  // Only runs while the stream is actually connected — otherwise the bar would
  // keep creeping forward on a dead connection, implying progress we can't see.
  useEffect(() => {
    if (!jobId) return;
    const id = setInterval(() => {
      if (!liveRef.current) return;
      const stage = stageRef.current;
      if (Date.now() - lastUpdateRef.current < 10_000) return;
      if (stage === 'Transcribing') {
        setFraction(f => Math.min(f + Math.min(0.002, (0.43 - f) / 20), 0.42));
      } else if (stage === 'Aligning Words') {
        setFraction(f => Math.min(f + Math.min(0.002, (0.60 - f) / 20), 0.59));
      } else if (stage === 'Speaker Diarization') {
        setFraction(f => Math.min(f + Math.min(0.002, (0.88 - f) / 20), 0.87));
      }
    }, 5_000);
    return () => clearInterval(id);
  }, [jobId]);

  return {
    currentStage,
    fraction,
    completed,
    error,
    connectionLost,
    stages: STAGES,
    audioDuration,
    cpuPct,
    ramGb,
    stageStartedAt,
    pipelineDecisions,
    partialTranscript,
  };
}
