# TekGPT Data Analyst: local Python service

This module extends the existing TekGPT Agent React/Vinext app. It does not replace its chat, tasks, Groq connection, Knowledge, memory, workflows or original sandboxed browser Python editor. Navigation is now **Data Analyst**; **Python editor** inside it preserves that editor.

## What is implemented

- Owner-isolated CSV datasets, original data and cleaned revisions, typed previews and calculated quality counts.
- Five controlled analytical tools: profile, chart, statistics, ML, preview_clean. No generated code, shell commands or arbitrary filesystem paths are accepted.
- Groq selects a controlled tool, Python executes it, then Groq interprets its recorded result. Per-dataset conversation context, explicit evidence IDs and revision numbers. A turn runs one operation; follow-up turns build on prior results. Existing Groq credentials stay on the Worker server. No streaming was added because the existing integration is non-streaming.
- Eight interactive Plotly chart types, column selectors, numeric validation, category grouping, filters/date ranges, sorting, aggregation, missing-value handling, duplication/editing/deletion, PNG downloads and report inclusion.
- EDA: missingness, duplicates, numeric descriptive statistics, categorical frequencies, variant counts, IQR/z-score flags, Pearson/Spearman matrices. AI interpretations are requested explicitly from recorded results.
- Cleaning preview/approval with stale-preview rejection; original CSV stays unchanged. Mean/median/mode/constant imputation, row/column removal, rename, conversion, trimming, filters and duplicates.
- Pearson/Spearman correlations, t intervals, Welch independent t-tests, classical ANOVA and chi-square with assumption diagnostics and warnings.
- Linear/multiple regression, RF regression; logistic/tree/RF/SVM classification; K-Means, DBSCAN and agglomerative clustering. XGBoost is optional. Preprocessing is fitted on training data only; reproducible random holdout. Regression, classification and clustering evaluation metrics and feature importance where available. Linear raw-unit equation is provided for all-numeric predictors; mixed features include transformed coefficients.
- PDF reports with static chart images and Excel workbooks with formatted, formula-safe cells. Reports include only executed analyses and distinguish revisions. AI interpretations are labelled. Source text and evidence fields are escaped; formula-looking user strings are exported as text.
- Server-side process jobs, two workers, 120-second hard execution limit. Job status persists in SQLite; jobs interrupted by a restart are marked interrupted when polled, not automatically resumed.

## Windows setup (Python 3.11+)

Extract the source ZIP. In PowerShell from its TekGPT-Agent folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\analyst_service\start-windows.ps1
```

The script creates `.analyst-venv`, installs dependencies and generates a private service token in `.analyst-service-token`. It listens on `127.0.0.1:8003`. Keep that PowerShell window running. Visit `http://127.0.0.1:8003/health` to check startup. Health is public and returns no dataset information; all analysis and download endpoints require both service authentication and an owner header.

## Connect the existing hosted Site

**Running the service locally alone does not connect the hosted app.** The Cloudflare Worker cannot call your computer's localhost. A trusted HTTPS tunnel or reverse proxy must reach port 8003 on your computer. Configure these **server-side Sites runtime variables**, then republish:

| Variable | Value |
| --- | --- |
| `ANALYST_SERVICE_URL` | Trusted HTTPS origin only, e.g. `https://your-analysis-host.example` (no path). |
| `ANALYST_SERVICE_TOKEN` | The exact generated service token from `.analyst-service-token`. Never commit or put it in chat. |

The same token is in the local Python process environment. The Worker forwards the authenticated account identity to Python, never a client-provided owner. CORS access is intentionally not enabled: browsers use the authenticated `/api/analyst` proxy. Keep the token private and restrict the public service route where your tunnel supports it. Disconnect by removing the Site runtime variables. Rotate the local token and Site token together.

The module deliberately shows **Python service not connected** and disables analytical controls until runtime variables exist. `configured` indicates configuration, not a health guarantee; unreachable services produce errors. Keep your computer awake while jobs run. The hosted agent, unrelated features and browser Python editor remain usable without this connection.

A concrete tunnel hostname and private runtime token must be configured before the hosted end-to-end acceptance flow can pass. This repository does not provision a tunnel, install software on your computer or pretend that a server is already connected.

## Other run options

```bash
python -m venv .analyst-venv
. .analyst-venv/bin/activate
pip install -r analyst_service/requirements.txt
# Set ANALYST_SERVICE_TOKEN to a random secret of at least 32 characters.
export ANALYST_DATA_DIR=./analyst-data
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
uvicorn analyst_service.app:app --host 127.0.0.1 --port 8003
```

Docker, from repository root:

```bash
docker build -f analyst_service/Dockerfile -t tekgpt-analyst .
docker run --rm -p 127.0.0.1:8003:8003 --env ANALYST_SERVICE_TOKEN --mount source=tekgpt-analysis,target=/data tekgpt-analyst
```

An optional XGBoost installation is `pip install xgboost` inside the Python environment. It is not installed by default; choosing it without installation produces a clear error.

## Storage and execution limits

Python stores `state.sqlite` and per-dataset revision CSVs/figures/reports in `ANALYST_DATA_DIR`. Back up that folder to retain data. The existing D1 schema is unchanged. Site per-user encryption and model preferences are reused.

CSV: UTF-8, <=5 MB, 1–100,000 rows, <=100 unique named columns; <=10 datasets/account. Analysis history: <=50 completed jobs/dataset; <=2 MB/result and <=6 MB recorded results/dataset. ML: <=20,000 rows, <=30 predictors; SVM/DBSCAN/hierarchical <=3,000 rows. Scatter <=10,000 rows; heatmap <=30 numeric columns. Category display is explicitly capped; exported chart data <=10,000 rows. Delete old results when reaching output limits.

These bounds support laptop-scale work, not distributed training. No GPU server, durable distributed scheduler, automatic model tuning, time-series validation, causal inference or automated statistical-method selection is claimed. Independence cannot be established automatically from a CSV. Python service data is never automatically sent to unrelated providers; previews and calculated result excerpts are sent to the selected Groq model when using AI Analyst.

## Tests

```bash
python -m pytest analyst_service/tests -q
node scripts/test-analyst.mjs
node scripts/test-agent.mjs
npx tsc --noEmit
npm run build
```

Python integration tests execute upload → analysis → cleaning approval → PDF/Excel export, validate numerical fixtures, all chart types, report image inclusion and spreadsheet formula protection, and deny cross-owner reads/jobs/downloads. Worker tests mock Groq and the service; they are not a live Groq certification. Production service connectivity and local Windows startup must be verified on the user's computer.

## File map

- `app/components/data-analyst.tsx`: dedicated tabbed workspace, uploads, AI chat, Plotly charts, analytical controls and reports.
- `app/components/workbench.tsx`, `app/page.tsx`, `app/globals.css`, `app/modules.d.ts`: integrate new module, preserve legacy editor and styles.
- `app/api/analyst/route.ts`: authenticated service proxy and Groq analytical orchestration.
- `analyst_service/app.py`: protected FastAPI API, SQLite metadata, revision storage, bounded background worker processes.
- `analyst_service/analysis.py`: controlled analytical implementations.
- `analyst_service/reports.py`: PDF/Excel generation from recorded evidence.
- `analyst_service/tests/test_analysis.py`, `scripts/test-analyst.mjs`: analytical, integration and proxy tests.
- `analyst_service/requirements.txt`, `Dockerfile`, `start-windows.ps1`: local service dependencies and launch.
- `package.json`, `package-lock.json`: add `plotly.js-dist-min` for interactive charts.
- `.gitignore`: exclude local environments, tokens and analysis data.
