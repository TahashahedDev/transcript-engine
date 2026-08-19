/**
 * Same-origin proxy to the FastAPI backend.
 *
 * Lets the browser talk only to the frontend's own origin (:9098) — the API
 * port (:9097) never needs to be reachable from outside the box. This is what
 * makes the app work on hosts (Vast.ai and similar) that only forward one
 * port: the backend stays on loopback, and this route, running server-side in
 * the same container, is the only thing that talks to it directly.
 *
 * Every request/response body is streamed straight through — never buffered —
 * so large audio uploads, artifact/bundle downloads, and the SSE progress
 * stream all pass through unchanged. See web/lib/apiBase.ts for the client
 * side of this (same-origin relative base by default).
 */

import type { NextRequest } from 'next/server';

export const runtime = 'nodejs';

const BACKEND_ORIGIN = `http://127.0.0.1:${process.env.TE_API_PORT ?? '9097'}`;

// Hop-by-hop / connection-specific headers: forwarding these verbatim between
// two different HTTP connections is incorrect (Content-Length in particular
// must be recomputed by whichever layer actually serializes the body, not
// copied from the other hop).
const STRIP_REQUEST_HEADERS = new Set(['host', 'connection', 'content-length']);
const STRIP_RESPONSE_HEADERS = new Set(['connection', 'transfer-encoding', 'content-encoding']);

function upstreamUrl(path: string[], search: string): string {
  const suffix = path.map(encodeURIComponent).join('/');
  return `${BACKEND_ORIGIN}/api/${suffix}${search}`;
}

async function proxy(req: NextRequest, path: string[]): Promise<Response> {
  const target = upstreamUrl(path, req.nextUrl.search);

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (!STRIP_REQUEST_HEADERS.has(key.toLowerCase())) headers.set(key, value);
  });

  // GET/DELETE never carry a body in this app (see lib/api.ts) — forwarding
  // `null` rather than a stream avoids the Node fetch duplex requirement
  // where it isn't needed.
  const body = req.method === 'GET' || req.method === 'DELETE' ? null : req.body;

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body,
      // Required by Node's fetch (undici) whenever `body` is a ReadableStream
      // — without it, a streamed request body throws synchronously.
      ...(body ? { duplex: 'half' as const } : {}),
      redirect: 'manual',
      cache: 'no-store',
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return Response.json(
      { detail: `Backend unreachable at ${BACKEND_ORIGIN}: ${message}` },
      { status: 502 },
    );
  }

  const responseHeaders = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!STRIP_RESPONSE_HEADERS.has(key.toLowerCase())) responseHeaders.set(key, value);
  });

  // `upstream.body` is passed straight through as the Response body — this is
  // what makes SSE (text/event-stream) and large file downloads stream
  // instead of buffering fully in memory before the client sees anything.
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders,
  });
}

type RouteContext = { params: Promise<{ path: string[] }> };

export async function GET(req: NextRequest, { params }: RouteContext): Promise<Response> {
  const { path } = await params;
  return proxy(req, path);
}

export async function POST(req: NextRequest, { params }: RouteContext): Promise<Response> {
  const { path } = await params;
  return proxy(req, path);
}

export async function DELETE(req: NextRequest, { params }: RouteContext): Promise<Response> {
  const { path } = await params;
  return proxy(req, path);
}
