## Frontend (React + Vite)

Development:

1. Install deps
```
pnpm install  # or npm install / yarn
```
2. Start dev server
```
pnpm run dev
```
3. Open http://localhost:5173

The dev proxy (vite.config.js) forwards /chat, /stats, /health to the FastAPI backend on :8000 so CORS headaches are avoided during local development.

Configure a different backend host:

```
VITE_API_URL=http://127.0.0.1:8000 pnpm run dev
```

Build production bundle:
```
pnpm run build
```
Output goes to `dist/` which you can serve behind any static server (NGINX, FastAPI StaticFiles, etc.).
