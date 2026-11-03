"use client";

import { FormEvent, useEffect, useState } from "react";
import HistoryPanel, { type History, type Statement } from "./history-panel";

type User = { id: number; email: string; customer_id: string };
type Indicators = Record<string, string | number | null>;
type Account = { account_id: string; account_type: string; balance: string; monthly_payment: string; apr_pct: string };
type Financials = { indicators: Indicators; accounts: Account[]; history: History; account_history: Statement[] };
type Scenario = { purchase_price: number; down_payment: number; apr_pct: number; term_months: number };
class ApiError extends Error { constructor(message: string, public status: number) { super(message); } }
function detailMessage(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const lines = detail.flatMap((entry): string[] => {
      if (!entry || typeof entry !== "object") return [];
      const node = entry as { msg?: unknown; loc?: unknown };
      const msg = typeof node.msg === "string" ? node.msg : "";
      const location = Array.isArray(node.loc)
        ? node.loc.filter((part): part is string | number => typeof part === "string" || typeof part === "number").join(".")
        : "";
      if (!msg) return [];
      return [location ? `${location}: ${msg}` : msg];
    });
    if (lines.length > 0) return lines.join(" | ");
  }
  return "Check your inputs and try again.";
}

async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST", credentials: "same-origin", cache: "no-store",
    headers: { "Content-Type": "application/json", "X-Workshop-CSRF": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.status === 204) return undefined as T;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(detailMessage(data.detail), response.status);
  return data;
}
const money = (value: unknown) => typeof value === "number" ? value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }) : "—";

export default function Home() {
  const [user, setUser] = useState<User | null>(null);
  const [financials, setFinancials] = useState<Financials | null>(null);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [scenario, setScenario] = useState<Scenario>({ purchase_price: 30000, down_payment: 5000, apr_pct: 8, term_months: 60 });
  const [result, setResult] = useState<Indicators | null>(null);
  const [answer, setAnswer] = useState("");
  function fail(exc: unknown) {
    if (exc instanceof ApiError && exc.status === 401) { setUser(null); setFinancials(null); setAnswer(""); setResult(null); }
    setError(exc instanceof Error ? exc.message : "Request failed");
  }
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const current = await api<User>("me");
        const data = await api<Financials>("financials");
        if (active) { setUser(current); setFinancials(data); }
      } catch (exc) {
        if (active && !(exc instanceof ApiError && exc.status === 401)) setError(exc instanceof Error ? exc.message : "API unavailable");
      } finally { if (active) setLoading(false); }
    })();
    return () => { active = false; };
  }, []);
  async function authenticate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError("");
    const form = event.currentTarget;
    const data = new FormData(form);
    try {
      const current = await api<User>(`auth/${mode}`, { email: data.get("email"), password: data.get("password") });
      form.reset(); setUser(current);
      setFinancials(await api<Financials>("financials"));
    } catch (exc) { fail(exc); } finally { setBusy(false); }
  }
  async function logout() {
    setBusy(true); setError("");
    try { await api("auth/logout", {}); setUser(null); setFinancials(null); setResult(null); setAnswer(""); }
    catch (exc) { fail(exc); } finally { setBusy(false); }
  }
  async function runScenario(explain: boolean) {
    setBusy(true); setError(""); setAnswer("");
    try {
      setResult(await api<Indicators>("scenario", scenario));
      if (explain) {
        const response = await api<{ answer: string; provider: string }>("narration", scenario);
        setAnswer(response.answer);
      }
    } catch (exc) { fail(exc); } finally { setBusy(false); }
  }
  const ind = financials?.indicators;
  return <main>
    <header><a className="brand" href="/">W<span>／</span> Financial Wellness Lab</a><span className="badge">PNC × UT Dallas · Workshop</span></header>
    <section className="intro"><p className="eyebrow">UNDERSTAND TODAY. EXPLORE TOMORROW.</p><h1>A clearer view of your<br />financial possibilities.</h1><p>Explore your synthetic household, test a vehicle purchase, and get an explanation grounded in calculated results.</p></section>
    <p className="notice">Synthetic data only · Educational prototype · Not banking, credit, investment, or lending advice.</p>
    {error && <div role="alert" className="error">{error}</div>}
    {loading ? <p role="status">Checking your session…</p> : !user ? <section className="auth card">
      <div><p className="eyebrow">YOUR WORKSHOP WORKSPACE</p><h2>{mode === "login" ? "Welcome back." : "Start exploring."}</h2><p>Each new registration receives its own synthetic customer profile. Use a fictional email address and a unique workshop password.</p><ul><li>Explore cash flow and reserves</li><li>Compare financing scenarios</li><li>Review a grounded explanation</li></ul></div>
      <form onSubmit={authenticate}><div className="tabs"><button type="button" aria-pressed={mode === "login"} onClick={() => { setMode("login"); setError(""); }}>Sign in</button><button type="button" aria-pressed={mode === "register"} onClick={() => { setMode("register"); setError(""); }}>Register</button></div>
        <label>Email<input name="email" type="email" required maxLength={254} autoComplete="username" placeholder="student@example.com" /></label>
        <label>Password<input name="password" type="password" required minLength={12} maxLength={128} autoComplete={mode === "login" ? "current-password" : "new-password"} /></label>
        <small>12–128 characters. No real banking credentials.</small>
        <button className="primary" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in to your workspace →" : "Create workshop account →"}</button>
      </form>
    </section> : <>
      <div className="account-bar"><div><h2>Your financial picture</h2><p>{user.email} · Synthetic profile {user.customer_id}</p></div><button disabled={busy} onClick={logout}>Sign out</button></div>
      <section className="metrics" aria-label="Financial indicators">
        <article className="card"><p>Monthly cash surplus</p><strong>{money(ind?.monthly_cash_surplus)}</strong><small>Income less modeled obligations</small></article>
        <article className="card"><p>Emergency reserves</p><strong>{typeof ind?.emergency_fund_months === "number" ? ind.emergency_fund_months.toFixed(2) : "—"}<em> months</em></strong><small>Checking + savings / monthly outflow</small></article>
        <article className="card"><p>Debt payment / income</p><strong>{ind?.debt_payment_to_income_pct ?? "—"}%</strong><small>Scheduled payments / modeled income</small></article>
      </section>
      <div className="workspace"><section className="card"><p className="eyebrow">EXPLORE A POSSIBILITY</p><h2>What if you bought a vehicle?</h2><p>Adjust the assumptions to see the effect on cash flow and liquid reserves.</p>
        <form onSubmit={event => { event.preventDefault(); void runScenario(false); }}><div className="fields">
          {([{ key: "purchase_price", label: "Purchase price ($)", max: 150000, min: 0, step: 100 }, { key: "down_payment", label: "Down payment ($)", max: scenario.purchase_price, min: 0, step: 100 }, { key: "apr_pct", label: "APR (%)", max: 35, min: 0, step: 0.1 }, { key: "term_months", label: "Loan term (months)", max: 120, min: 6, step: 1 }] as const).map(field => <label key={field.key}>{field.label}<input disabled={busy} type="number" required min={field.min} max={field.max} step={field.step} value={scenario[field.key]} onChange={event => { setScenario({ ...scenario, [field.key]: Number(event.target.value) }); setResult(null); setAnswer(""); }} /></label>)}
        </div><div className="actions"><button className="primary" disabled={busy}>Calculate scenario</button><button disabled={busy} type="submit" onClick={event => { const form = event.currentTarget.form; if (form?.reportValidity()) { event.preventDefault(); void runScenario(true); } }}>Calculate + explain</button></div></form>
        {busy && <p role="status">Working… local model startup may take a few minutes.</p>}
        {result && <div className="results" aria-live="polite"><p>New monthly payment <b>{money(result.new_monthly_payment)}</b></p><p>Projected monthly surplus <b>{money(result.projected_monthly_surplus)}</b></p><p>Liquid assets after down payment <b>{money(result.remaining_liquid_assets)}</b></p><p>Projected reserve months <b>{result.projected_emergency_fund_months ?? "—"}</b></p><small>{result.caveat}</small></div>}
      </section><aside className="card"><p className="eyebrow">FROM EVIDENCE TO EXPLANATION</p><h2>Grounded considerations</h2><p className="narrative" aria-live="polite">{answer || "Choose Calculate + explain to review your scenario. Calculations come from the financial tools; narration uses the backend’s configured provider."}</p><small>Review model interpretations against the figures. A cash surplus is not proof of actual savings.</small></aside></div>
      <section className="card"><h2>Your synthetic accounts</h2><div className="table-wrap"><table><thead><tr><th>Account type</th><th>Balance</th><th>Monthly payment</th><th>APR</th></tr></thead><tbody>{financials?.accounts.map((account, index) => <tr key={`${account.account_type}-${index}`}><td>{account.account_type.replaceAll("_", " ")}</td><td>{money(Number(account.balance))}</td><td>{money(Number(account.monthly_payment))}</td><td>{Number(account.apr_pct).toFixed(2)}%</td></tr>)}</tbody></table></div><p><small>{ind?.financial_data_scope}</small></p></section>
      {financials && <HistoryPanel accounts={financials.accounts} history={financials.history} statements={financials.account_history} onExpired={() => { setUser(null); setFinancials(null); setAnswer(""); setResult(null); setError("Session expired; sign in again."); }} />}
    </>}
    <footer>Built for learning · Synthetic balances are partial · No transactions or credit decisions</footer>
  </main>;
}
