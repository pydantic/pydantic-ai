# Plan: Vertex server deadline vs. client timeout (refs #11902)

Draft plan for discussion before implementation, per CONTRIBUTING ("Propose a plan before building").
Every mechanism below was verified against `main` @ `6f3585c` and the `google-genai` SDK source on 2026-10-31.

## Verified mechanism chain

1. `BaseGoogleProvider._build_http_options` pins `HttpOptions.timeout` from the client's read timeout:
   `timeout_seconds = http_client.timeout.read or DEFAULT_HTTP_TIMEOUT` (`pydantic_ai_slim/pydantic_ai/providers/google.py:106`).
   This pin is deliberate — without it `google-genai` passes `timeout=None` into httpx per request, which disables
   the client's own timeout entirely (the SDK treats an explicit `None` as "no timeout", not "use the client default").
2. The SDK turns that single value into **both** the local request timeout and the server deadline:
   `_api_client.py` calls `populate_server_timeout_header(headers, timeout_in_seconds)`, which sets
   `X-Server-Timeout: ceil(seconds)` unless the header is already present, and builds
   `HttpRequest(..., timeout=timeout_in_seconds)`. There is no public way to send one without the other.
3. Consequences for a user-supplied client:
   - client built without `timeout=` → httpx default `read=5.0` → **every Vertex generation is aborted server-side
     after 5 s** with a 504 `DEADLINE_EXCEEDED`, even though the caller never asked for a deadline;
   - client built with `read=None` (no read timeout wanted) → falls back to `DEFAULT_HTTP_TIMEOUT` (600 s) as the
     server deadline — also not what the caller asked for;
   - a provider-owned client is fine: `_create_google_httpx2_client` uses a 20-minute timeout
     (`providers/google.py:33-39`, Google's own long-decode recommendation).
4. The 504 `DEADLINE_EXCEEDED` is classified as a provider fault. `ModelServerError` deliberately covers
   "an HTTP 500, 502 or 504, or a gRPC INTERNAL or DEADLINE_EXCEEDED sent by the provider" (`exceptions.py`), and the
   default `fallback_on` includes it — so a deadline the caller's own client imposed triggers model fallback.
   Logfire works around this with its own checks (per the issue).

## Two separable problems

- **P1 — the caller's client timeout becomes a server deadline.** Behavior bug, affects user-supplied clients only.
- **P2 — `DEADLINE_EXCEEDED` attribution.** Today it is always a provider fault; when the deadline came from the
  request itself it should read as a timeout, so fallback logic doesn't misfire.

## Options

### P1

- **A′. Substitute our default when the user's client has an untouched httpx default.** If the user-supplied client's
  read timeout equals the httpx default (5.0 s) — or is `None` — use the provider-owned default for `HttpOptions.timeout`
  and emit a one-time warning explaining the server-deadline side effect. Smallest diff, reversible, testable; the
  caveat is that we cannot distinguish "default 5 s" from "explicitly chose 5 s", so those callers lose their 5 s
  server deadline (they keep it locally) and get a warning telling them how to opt back in.
- **B. Decouple local timeout from server deadline.** Only pin `HttpOptions.timeout` when we own the client; for
  user-supplied clients, enforce their timeout at the transport layer (wrap the client so the read timeout is applied
  locally while `HttpOptions.timeout` stays unset). Cleanest semantics — "a user-supplied client means the caller owns
  timeouts" — but it is a bigger change and likely wants an upstream `google-genai` ask, since the SDK couples the two
  values in `populate_server_timeout_header`.
- **C. Explicit knob.** e.g. `GoogleProvider(..., server_deadline=...)` or an `HttpOptions` passthrough. Most honest
  API, but adds public surface for something that arguably shouldn't exist by default. (Maintainer call.)

### P2

- **D. Attribute the deadline where possible**, modeled on the xAI provider's existing approach:
  `models/xai.py::_grpc_transport_error` maps a gRPC `DEADLINE_EXCEEDED` whose `details` match
  `_GRPC_CLIENT_DEADLINE` to `ModelTimeoutError`, and leaves server-sent errors alone. For Vertex the response body
  carries `DEADLINE_EXCEEDED` with the deadline duration, but the response does not echo *which* value produced it,
  so attribution needs a rule (e.g. only when the request carried `X-Server-Timeout` derived from the client, which
  requires tracking what we sent).
- **Or keep P2 as-is for now.** After P1, a `DEADLINE_EXCEEDED` from a provider-owned client means a genuine
  provider-side or 20-minute deadline, which the current `ModelServerError` classification handles correctly.

## Recommendation

A′ for P1 as the first step (small, testable, safe to revert), plus filing the upstream `google-genai` decoupling
request that option B needs. Revisit P2 (option D) once A′ is in and there is real Vertex evidence to design the
attribution rule against — guessing the rule without traffic data is how we'd end up misclassifying genuine provider
outages.

## Open questions for maintainers

1. Blast radius: the SDK sends `X-Server-Timeout` on every request through this client path — does the Gemini API
   (developer AI Studio endpoint) honor it, or only Vertex? This decides whether P1 changes affect Gemini API users.
2. Is "user-supplied client ⇒ the caller owns timeouts" the intended contract across providers? If yes, B is the
   right end state; if not, C's knob is needed.
3. For callers who set `read=None` (currently → 600 s server deadline): intended, or should they get no server
   deadline at all?

## Test plan (what I can verify locally)

- Unit tests with a stubbed `google-genai` client capturing the `HttpOptions` we pass: provider-owned client → 20 min;
  user client with default timeout → substituted default + warning (A′); user client with explicit `timeout=600` →
  600 s preserved unchanged; user client with `read=None` → documented behavior from the open question.
- If P2/D proceeds: mapping tests mirroring `models/xai.py`'s details-based discrimination, plus a regression that
  genuine server 504s still classify as `ModelServerError`.
- I do not have Vertex access: end-to-end header behavior and the 504 body shape come from the SDK source and the
  issue report, and need confirmation from someone with a Vertex project.
