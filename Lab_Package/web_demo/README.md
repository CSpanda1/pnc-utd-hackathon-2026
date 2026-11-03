# Next.js + FastAPI workshop

A second front end for the financial wellness lab: registration, sign-in/out, a customer dashboard, 24-month account/transaction history, vehicle simulation, and grounded narration using the existing Python tools. Everything uses synthetic CSV data. No real customer data or banking credentials.

```text
Browser → Next.js /api/* route handler → FastAPI → SQLite identity/session store
                                             → financial_tools.py → synthetic CSVs
                                             → llm_demo.py → Hugging Face or Foundry Local model
```

The backend assigns the next available synthetic profile at registration, up to 30 users. A user cannot select a customer ID. All financial endpoints derive that ID from the authenticated session. Registrations persist across restarts.

## Start locally

Run both services on the same machine. The frontend requires Node.js 20.9+; use Node 24 with the pinned pnpm 11 toolchain. Python 3.10+ is required for FastAPI. Use **WSL for both services** when demonstrating the previously downloaded Hugging Face models. For the default evidence-only mode, Windows works too. You need two terminals.

### Terminal 1: FastAPI

From this `web_demo` directory, on WSL/Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The API stores its SQLite database in ignored `backend/state/workshop.sqlite3`. API documentation: <http://127.0.0.1:8000/docs>. The docs UI is for inspecting contracts; mutation requests also need the origin and CSRF header described below.

### Terminal 2: Next.js

On WSL/Linux, the launcher fetches Node 24 and pnpm 11.25.0 into npm's cache, even when the system Node is older. It does not replace system Node:

```bash
cd frontend
bash scripts/frontend.sh install --frozen-lockfile
cp .env.example .env.local  # Skip if you already have .env.local
bash scripts/frontend.sh dev
```

The first invocation needs internet access to fetch the runtime and packages. Subsequent runs reuse the cache. To build or type-check, use `bash scripts/frontend.sh build` or `bash scripts/frontend.sh typecheck`.

On Windows with Node 24 and pnpm installed:

```powershell
cd frontend
pnpm install --frozen-lockfile
Copy-Item .env.example .env.local  # Skip if it already exists
pnpm dev
```

Use pnpm consistently; the project pins its version in `package.json` and includes `pnpm-lock.yaml`. Do not run `npm install` over a pnpm-managed `node_modules` directory. Windows and WSL need separate dependency installations because native packages and links differ. When switching operating systems, move the existing `node_modules` and `.next` aside before installing in the other environment. Keep `.env.local` and source files.

If npm reports `Cannot read properties of null (reading 'isDescendantOf')` with `.pnpm` paths, stop any frontend process and, from `frontend` in WSL, preserve the mixed installation before reinstalling:

```bash
mv node_modules "node_modules.windows-backup-$(date +%s)"
if [ -d .next ]; then mv .next ".next.windows-backup-$(date +%s)"; fi
bash scripts/frontend.sh install --frozen-lockfile
bash scripts/frontend.sh dev
```

Do not repeat the move if the old install has already been moved. `.nvmrc` also declares Node 24 for users who prefer their own version manager.

Open **http://localhost:3000**. Use this hostname consistently: origin checks intentionally distinguish `localhost` from `127.0.0.1`. Register with a fictional email such as `student@example.com` and a unique 12–128 character password. Registration signs you in immediately. Select **Calculate scenario**, then **Calculate + explain**.

### Enable the cached local model

Run FastAPI from the existing WSL inference environment instead of the minimal web `.venv`. Install `backend/requirements.txt` into that environment. Keep its working CUDA PyTorch installation.

```bash
# From web_demo, after activating your WSL model environment:
export HF_HUB_OFFLINE=1
export HF_MODEL=Qwen/Qwen3-VL-2B-Instruct
export LLM_PROVIDER=huggingface
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

If inference dependencies are missing, see `../requirements-hf.txt` and the parent README. The first explanation loads cached weights; the API allows one narration at a time, and the Next.js proxy waits up to five minutes. A timed-out browser request does not cancel a running GPU computation. Keep **one API worker** to avoid duplicated GPU models. `LLM_PROVIDER=none` is the default and returns evidence without requiring a model. You can set `LLM_PROVIDER=huggingface`, `LLM_PROVIDER=foundry`, or `LLM_PROVIDER=openai` on the backend process only; the browser cannot choose providers or send its own evidence.

## API contract and security walkthrough

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/auth/register` | `{email,password}`; allocate synthetic profile and session |
| POST | `/auth/login` | `{email,password}`; rotate browser session |
| POST | `/auth/logout` | `{}`; revoke session and clear cookie |
| GET | `/me` | Authenticated user and assigned profile |
| GET | `/financials` | Authenticated user's indicators and accounts |
| GET | `/transactions` | Authenticated user's paginated ledger; optional account/date filters |
| POST | `/scenario` | Validated vehicle assumptions, calculated on server |
| POST | `/narration` | Recalculate evidence for the signed-in user, then explain |

The browser prefixes each path with `/api`. `/financials` additionally returns the authenticated customer's monthly history summary and account statements. `/transactions` accepts only `account_id`, `start_date`, `end_date`, `limit` (1–200, default 50), and `offset` (nonnegative). Dates are inclusive ISO dates; the backend rejects accounts outside the signed-in customer's scope. Example: `/api/transactions?account_id=A-C001-CHECKING&start_date=2026-10-01&end_date=2026-10-31&limit=50&offset=0`. The account ID must belong to the signed-in user.

The history panel shows November 2024–October 2026, with monthly paychecks, spending, debt payments, fees, interest, liquidity, and explicit external calibration flows. Account selection filters statements and ledger entries; date filters apply to ledger entries only. Asset/liability balances are positive and signed changes describe movement in that account. See `../data/README.md` for detailed accounting conventions. Historical amounts are generated activity, separate from snapshot assumptions used in the scenario calculator.

A scenario body is:

```json
{"purchase_price":30000,"down_payment":5000,"apr_pct":8,"term_months":60}
```

- Argon2id hashes protect passwords. SQL uses bound parameters.
- Random 256-bit opaque session tokens are stored only as SHA-256 hashes in SQLite. Browser cookies are HttpOnly, SameSite=Strict, host-only, and expire after one hour. Login rotates the current browser session; logout revokes it server-side. Tokens are never stored in localStorage or returned in JSON.
- Both Next.js and FastAPI check the exact `Origin` plus `X-Workshop-CSRF: 1` on mutations. The frontend sends these headers automatically. CORS is not enabled: browser calls go through the same-origin proxy. A CSRF header is an additional browser-origin control, not an API credential.
- The proxy permits only known paths/methods and whitelisted transaction query parameters, caps request bodies at 4 KiB, and never accepts an upstream URL from the client. `BACKEND_URL` is server-only. Authentication responses and financial responses are not cached.
- User-to-customer authorization is enforced on the backend. Extra body fields such as `customer_id`, `provider`, or `evidence` are rejected. Transaction account filters must belong to the session's customer; client-supplied customer query parameters are rejected.
- SQLite-backed rate limits apply to authentication and narration. Login failures share the same message. Email-based limits can temporarily lock a targeted email; this is a teaching tradeoff. All users share a generous peer-IP limit behind the local Next.js proxy. Do not trust client-provided forwarding headers.
- Scenario inputs have bounded finite numbers, an integer term, and a down-payment check. Narration consumes server-calculated evidence and runs under a single-process inference lock.

For a direct API demonstration, send `Origin: http://localhost:3000`, `X-Workshop-CSRF: 1`, and `Content-Type: application/json` on POST requests and preserve the session cookie. The test suite provides executable examples.

## Workshop demonstration sequence

1. Register Alice; show the assigned C001 dashboard.
2. In a separate private browser session, register Bob; show C002.
3. In browser developer tools, inspect `/api/financials` and its HttpOnly session cookie. The request body carries no customer ID.
4. Try adding `customer_id` to a scenario request; expect HTTP 422. Call `/api/financials` while signed out; expect HTTP 401. Try a POST with a foreign Origin; expect HTTP 403.
5. Calculate a zero-APR scenario and compare the payment with principal / months.
6. Generate an explanation; compare every number and interpretation with deterministic evidence. Synthetic output may still contain unsupported interpretations.
7. Sign out; replay the old cookie and observe HTTP 401.

## Verification

From `web_demo`, with the backend environment active:

```bash
python -m unittest backend.test_api -v
```

From `frontend`:

```bash
bash scripts/frontend.sh typecheck
bash scripts/frontend.sh build
```

On Windows with Node 24 and pnpm installed, use `pnpm typecheck` and `pnpm build`.

The API tests use temporary databases and mocked narration. They cover missing/expired/revoked sessions, password hashing, cookie flags, CSRF/origin rejection, session rotation, profile isolation, allocation limits, brute-force limits, numeric validation, and server-generated narration evidence.

## Configuration and deployment boundaries

| Service | Variable | Default |
| --- | --- | --- |
| Next.js | `BACKEND_URL` | `http://127.0.0.1:8000` |
| Next.js | `APP_ORIGIN` | `http://localhost:3000` |
| FastAPI | `FRONTEND_ORIGIN` | `http://localhost:3000` |
| FastAPI | `COOKIE_SECURE` | `0`, permitted only for localhost development |
| FastAPI | `DATABASE_PATH` | `backend/state/workshop.sqlite3` |
| FastAPI | `LLM_PROVIDER` | `none`; optionally `huggingface`, `foundry`, or `openai` |
| FastAPI | `FOUNDRY_BASE_URL` | `http://127.0.0.1:5273/v1` |
| FastAPI | `FOUNDRY_MODEL` | `qwen2.5-0.5b` |
| FastAPI | `FOUNDRY_API_KEY` | Optional; defaults to `foundry-local` |
| FastAPI | `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |
| FastAPI | `OPENAI_MODEL` | `gpt-5.4` (override to `gpt-4o` / `gpt-4.1` as needed) |

For an HTTPS demonstration set both origin settings to the same exact HTTPS origin and `COOKIE_SECURE=1`. Keep FastAPI private to the Next.js server; use TLS for any network hop that leaves the machine. Place a request-size/rate-limiting reverse proxy in front of the API for nonlocal hosting. This example's UI uses a teaching CSP with inline scripts permitted; a deployed app should use a nonce-based CSP.

This is a workshop reference, not a production identity system. It intentionally omits email verification, password reset, MFA, federated SSO, account deletion, and security audit infrastructure. Use an established OIDC provider and managed persistence for a production deployment. There is no real account linking: registering grants an arbitrary synthetic profile for the exercise. SQLite and the model lock target a single API process. To reset the classroom, stop the API and remove only its ignored workshop SQLite file; this erases registrations and sessions.

Implementation references: [Next.js route handlers](https://nextjs.org/docs/app/api-reference/file-conventions/route), [FastAPI password hashing](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/).
