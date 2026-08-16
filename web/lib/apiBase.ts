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
 *   2. Same host as the page, on the API port. Correct for the normal
 *      deployment where both services run on one box: open the UI at
 *      http://<host>:9098 and the API is found at http://<host>:9097,
 *      whether <host> is localhost or a rented GPU's public IP.
 *   3. localhost fallback — only reached during SSR/prerender, where there is
 *      no window and no request is actually issued to the API.
 */

const DEFAULT_API_PORT = '9097';

export function resolveApiBase(): string {
  const configured = process.env.NEXT_PUBLIC_API_URL;
  if (configured) return configured.replace(/\/+$/, '');

  if (typeof window !== 'undefined') {
    const port = process.env.NEXT_PUBLIC_API_PORT ?? DEFAULT_API_PORT;
    return `${window.location.protocol}//${window.location.hostname}:${port}`;
  }

  return `http://localhost:${DEFAULT_API_PORT}`;
}
