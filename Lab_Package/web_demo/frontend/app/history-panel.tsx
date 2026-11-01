"use client";

import { useEffect, useState } from "react";

export type HistoryAccount = { account_id: string; account_type: string };
type Month = { month: string; paycheck_income: number; spending: number; debt_payments: number; debt_interest: number; fees: number; asset_interest: number; closing_liquid_assets: number; closing_assets: number; closing_debts: number; external_transfers_in: number; external_transfers_out: number };
export type History = { history_start: string; as_of_date: string; transaction_count: number; months: Month[]; definitions: string };
export type Statement = { account_id: string; account_type: string; month: string; opening_balance: number; closing_balance: number; balance_increases: number; balance_decreases: number; transaction_count: number };
type Transaction = { transaction_id: string; account_id: string; account_type: string; posted_date: string; transaction_type: string; category: string; amount: number; balance_change: number; balance_after: number; flow_class: string; transfer_id: string };
type Page = { items: Transaction[]; total: number; limit: number; offset: number };
const usd = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD" });

export default function HistoryPanel({ accounts, history, statements, onExpired }: { accounts: HistoryAccount[]; history: History; statements: Statement[]; onExpired: () => void }) {
  const [account, setAccount] = useState("");
  const [from, setFrom] = useState(history.history_start);
  const [through, setThrough] = useState(history.as_of_date);
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    if (!from || !through || from > through) { setError("Choose a valid date range."); setPage(null); setLoading(false); return; }
    setLoading(true); setError(""); setPage(null);
    const query = new URLSearchParams({ start_date: from, end_date: through, limit: "50", offset: String(offset) });
    if (account) query.set("account_id", account);
    (async () => {
      try {
        const response = await fetch(`/api/transactions?${query}`, { cache: "no-store", credentials: "same-origin", signal: controller.signal });
        const result = await response.json();
        if (response.status === 401) { onExpired(); return; }
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Transaction request failed");
        setPage(result);
      } catch (exc) { if (!controller.signal.aborted) setError(exc instanceof Error ? exc.message : "Request failed"); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  // onExpired only clears the parent session, and is excluded to avoid fetching on each render.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [account, from, through, offset]);
  const selected = statements.filter(row => !account || row.account_id === account);
  const largest = Math.max(1, ...history.months.flatMap(month => [month.paycheck_income, month.spending]));
  return <section className="card history"><p className="eyebrow">24 MONTHS OF SYNTHETIC ACTIVITY</p><h2>Account history and transactions</h2>
    <p>{history.history_start} through {history.as_of_date} · {history.transaction_count.toLocaleString()} ledger entries</p>
    <p><small>{history.definitions}</small></p>
    <details open><summary>Monthly household income and spending</summary><p><small>Teal: paychecks. Orange: spending. Debt payments are transfers and appear separately in the table.</small></p>
      <div className="month-bars" role="img" aria-label="Monthly generated paychecks and spending; exact values are in the following table">{history.months.map(month => <div key={month.month} title={`${month.month}: paychecks ${usd(month.paycheck_income)}, spending ${usd(month.spending)}`}><div className="bars"><i style={{ height: `${100 * month.paycheck_income / largest}%` }} /><i style={{ height: `${100 * month.spending / largest}%` }} /></div><span>{month.month.slice(2)}</span></div>)}</div>
      <div className="table-wrap"><table><thead><tr><th>Month</th><th>Paychecks</th><th>Spending</th><th>Debt payments</th><th>Debt interest</th><th>Fees</th><th>Liquid balance</th><th>External in</th><th>External out</th></tr></thead><tbody>{history.months.map(month => <tr key={month.month}><td>{month.month}</td><td>{usd(month.paycheck_income)}</td><td>{usd(month.spending)}</td><td>{usd(month.debt_payments)}</td><td>{usd(month.debt_interest)}</td><td>{usd(month.fees)}</td><td>{usd(month.closing_liquid_assets)}</td><td>{usd(month.external_transfers_in)}</td><td>{usd(month.external_transfers_out)}</td></tr>)}</tbody></table></div>
    </details>
    <div className="history-filters"><label>Account<select value={account} onChange={event => { setAccount(event.target.value); setOffset(0); }}><option value="">All accounts</option>{accounts.map(a => <option key={a.account_id} value={a.account_id}>{a.account_type.replaceAll("_", " ")} · {a.account_id}</option>)}</select></label>
      <label>From<input type="date" value={from} onChange={event => { setFrom(event.target.value); setOffset(0); }} /></label><label>Through<input type="date" value={through} onChange={event => { setThrough(event.target.value); setOffset(0); }} /></label></div>
    <details><summary>Monthly statements {account ? "for selected account" : "for all accounts"}</summary><div className="table-wrap"><table><thead><tr><th>Account</th><th>Month</th><th>Opening</th><th>Increases</th><th>Decreases</th><th>Closing</th></tr></thead><tbody>{selected.map(row => <tr key={`${row.account_id}-${row.month}`}><td>{row.account_type.replaceAll("_", " ")}</td><td>{row.month}</td><td>{usd(row.opening_balance)}</td><td>{usd(row.balance_increases)}</td><td>{usd(row.balance_decreases)}</td><td>{usd(row.closing_balance)}</td></tr>)}</tbody></table></div><small>Statement scope: full 24 months for the selected account. From/Through filters apply to transaction entries below.</small></details>
    <h3>Transaction entries</h3>{error && <p role="alert" className="error">{error}</p>}{loading && <p role="status">Loading transactions…</p>}
    <div className="table-wrap"><table><thead><tr><th>Date</th><th>Account</th><th>Event</th><th>Category</th><th>Amount</th><th>Balance change</th><th>Balance after</th><th>Flow</th><th>Transfer link</th></tr></thead><tbody>{page?.items.map(row => <tr key={row.transaction_id}><td>{row.posted_date}</td><td>{row.account_type.replaceAll("_", " ")}</td><td>{row.transaction_type.replaceAll("_", " ")}</td><td>{row.category.replaceAll("_", " ")}</td><td>{usd(row.amount)}</td><td>{usd(row.balance_change)}</td><td>{usd(row.balance_after)}</td><td>{row.flow_class.replaceAll("_", " ")}</td><td>{row.transfer_id || "—"}</td></tr>)}</tbody></table></div>
    {page && <div className="actions"><button disabled={loading || offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous</button><span aria-live="polite">{page.total === 0 ? "No matching entries" : `${offset + 1}–${Math.min(offset + 50, page.total)} of ${page.total}`}</span><button disabled={loading || offset + 50 >= page.total} onClick={() => setOffset(offset + 50)}>Next</button></div>}
    <p><small>Positive balance changes increase the asset or liability. Negative changes decrease it. Interest on savings, CDs, and IRAs is a simplified synthetic accrual, not an observed investment return.</small></p>
  </section>;
}
