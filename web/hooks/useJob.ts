'use client';

import { useEffect, useRef, useState } from 'react';
import { getJob } from '@/lib/api';
import type { Job } from '@/lib/types';

// Poll every 8s while running/queued — SSE handles real-time progress.
// This is only a fallback to detect server restarts and get final metrics.
const POLL_INTERVAL_MS = 8_000;
const ERROR_RETRY_MS = 10_000;

export function useJob(jobId: string | null, sseCompleted?: boolean) {
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const activeRef = useRef(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const fetchRef = useRef<(() => Promise<void>) | null>(null);

  useEffect(() => {
    if (!jobId) return;

    activeRef.current = true;

    async function poll() {
      try {
        const j = await getJob(jobId!);
        if (!activeRef.current) return;
        setJob(j);
        setLoading(false);
        if (j.status === 'completed' || j.status === 'failed') return;
        timerRef.current = setTimeout(poll, POLL_INTERVAL_MS);
      } catch {
        if (!activeRef.current) return;
        timerRef.current = setTimeout(poll, ERROR_RETRY_MS);
      }
    }

    fetchRef.current = poll;
    poll();

    return () => {
      activeRef.current = false;
      clearTimeout(timerRef.current);
    };
  }, [jobId]);

  // When SSE signals completion, do one immediate fetch to get final metrics
  // (artifacts list, word count, speaker count) without waiting for the next poll.
  useEffect(() => {
    if (!sseCompleted || !fetchRef.current) return;
    clearTimeout(timerRef.current);
    fetchRef.current();
  }, [sseCompleted]);

  return { job, loading };
}
