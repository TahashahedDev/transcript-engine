/**
 * Resolves the backend API base URL.
 *
 * Why this isn't just an env var: NEXT_PUBLIC_* values are inlined into the
 * client bundle at *build* time, and the bundle runs in the visitor's browser.
 * A baked-in "http://localhost:9097" therefore resolves to the visitor's own
 * machine, not the server — so a build made on a remote GPU host (Vast.ai and
 * similar) would ship a frontend that cannot reach its own backend, and
 * editing .env.local after the build would have no effect.
 *
 * Resolution order:
 *   1. NEXT_PUBLIC_API_URL — explicit override; use when the API lives on a
 *      different host/domain than the UI (separate domains, reverse proxy).
 *   2. Same-origin relative base ('' ). The browser never talks to the API
 *      port directly — requests to `${apiBase()}/api/...` resolve to
 *      `/api/...` on the page's own origin, which app/api/[...path]/route.ts
 *      proxies server-side to http://127.0.0.1:9097. This is what lets the
 *      API stay unexposed on hosts (like Vast.ai) that only forward the
 *      frontend's port: the browser only ever needs to reach :9098.
 *   3. localhost fallback — only reached during SSR/prerender, where there is
 *      no window and no request is actually issued to the API.
 */

const DEFAULT_API_PORT = '9097';

export function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL;
  if (configured) return configured.replace(/\/+$/, '');

  if (typeof window !== 'undefined') {
    return '';
  }

  return `http://localhost:${DEFAULT_API_PORT}`;
}
