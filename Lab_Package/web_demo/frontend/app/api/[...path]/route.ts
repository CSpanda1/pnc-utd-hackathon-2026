import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
const routes: Record<string, string> = {
  "auth/register": "POST", "auth/login": "POST", "auth/logout": "POST",
  me: "GET", financials: "GET", transactions: "GET", scenario: "POST", narration: "POST",
};

function isAllowedOrigin(actualOrigin: string | null, configuredOrigin: string): boolean {
  if (!actualOrigin) return false;
  if (actualOrigin === configuredOrigin) return true;

  const loopbackHosts = new Set(["localhost", "127.0.0.1", "::1"]);
  try {
    const actual = new URL(actualOrigin);
    const configured = new URL(configuredOrigin);
    return (
      actual.protocol === configured.protocol &&
      actual.port === configured.port &&
      loopbackHosts.has(actual.hostname) &&
      loopbackHosts.has(configured.hostname)
    );
  } catch {
    return false;
  }
}

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const path = (await context.params).path.join("/");
  if (!Object.hasOwn(routes, path)) return Response.json({ detail: "Not found" }, { status: 404 });
  if (request.method !== routes[path]) return Response.json({ detail: "Method not allowed" }, { status: 405 });
  const origin = process.env.APP_ORIGIN || "http://localhost:3000";
  if (
    request.method === "POST" &&
    (!isAllowedOrigin(request.headers.get("origin"), origin) || request.headers.get("x-workshop-csrf") !== "1")
  ) {
    return Response.json({ detail: "Origin or CSRF header rejected" }, { status: 403 });
  }
  const queryKeys = new Set(["account_id", "start_date", "end_date", "limit", "offset"]);
  if (request.nextUrl.search && (path !== "transactions" || [...request.nextUrl.searchParams.keys()].some(key => !queryKeys.has(key)))) {
    return Response.json({ detail: "Unsupported query parameters" }, { status: 400 });
  }
  const headers = new Headers({ "Content-Type": "application/json" });
  for (const name of ["cookie", "x-workshop-csrf"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  // Forward a canonical origin to match backend FRONTEND_ORIGIN checks.
  headers.set("origin", origin);
  let body: string | undefined;
  if (request.method === "POST") {
    if (request.headers.get("content-type")?.split(";")[0] !== "application/json") return Response.json({ detail: "JSON required" }, { status: 415 });
    // Bound streamed bodies too; do not rely on the Content-Length header.
    const reader = request.body?.getReader();
    const chunks: Uint8Array[] = [];
    let size = 0;
    if (reader) {
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        size += chunk.value.length;
        if (size > 4096) { await reader.cancel(); return Response.json({ detail: "Request too large" }, { status: 413 }); }
        chunks.push(chunk.value);
      }
    }
    body = Buffer.concat(chunks).toString("utf8");
  }
  try {
    const upstream = await fetch(`${process.env.BACKEND_URL || "http://127.0.0.1:8000"}/${path}${request.nextUrl.search}`, {
      method: request.method, headers, body, cache: "no-store", redirect: "error",
      signal: AbortSignal.timeout(path === "narration" ? 300000 : 15000),
    });
    const outgoing = new Headers({ "Cache-Control": "no-store", "Content-Type": "application/json" });
    for (const cookie of upstream.headers.getSetCookie()) outgoing.append("Set-Cookie", cookie);
    const retry = upstream.headers.get("retry-after");
    if (retry) outgoing.set("Retry-After", retry);
    return new Response(upstream.body, { status: upstream.status, headers: outgoing });
  } catch {
    return Response.json({ detail: "API unavailable or timed out. Please retry." }, { status: 502 });
  }
}
export { proxy as GET, proxy as POST };
