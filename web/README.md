# Web UI

Next.js frontend for Transcript Engine: upload a recording, watch progress live,
read the transcript, download the artifacts.

Normally you do not run this directly — `bash start.sh` at the repo root starts
the API and this UI together, and builds it on first run.

## Running it on its own

```bash
npm ci
npm run dev      # http://localhost:9098, hot reload
```

The API must be running too (`bash start.sh --backend-only`).

| Command | |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build |
| `npm start` | Serve the production build |
| `npm run lint` | ESLint |
| `npm run typecheck` | `tsc --noEmit` |

## How it finds the API

At runtime, from the address you loaded the page on — see `lib/apiBase.ts`.
Nothing is baked in at build time, so the same build works on `localhost` and on
a rented GPU box reached by IP.

`NEXT_PUBLIC_API_URL` overrides that when the API lives on a different host
(see `.env.local.example`). Setting it is a build-time choice: `NEXT_PUBLIC_*`
values are inlined into the client bundle, so changing it needs a rebuild.

## Layout

```
app/          routes: upload (/), job view, diagnostics, setup
components/   feature components; ui/ holds the shadcn primitives
hooks/        useJob (polling), useProgress (SSE)
lib/          API client, types, small helpers
```
