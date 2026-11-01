# Plan: Vertex server deadline vs. client timeout (refs #11902)

Discussion-only plan; no runtime implementation is included.
The original analysis used `main` at `6f3585c`. This revision corrects the timeout assumptions and
records [the maintainer's direction](https://github.com/pydantic/pydantic-ai/issues/11902#issuecomment-6536168151).
The maintainers will implement the fix themselves.

## Verified mechanism

1. `BaseGoogleProvider._build_http_options` pins `HttpOptions.timeout` from
   `http_client.timeout.read or DEFAULT_HTTP_TIMEOUT` (`providers/google.py`). This preserves the
   anti-hang fix: leaving the SDK timeout unset passes `timeout=None` to httpx for individual requests,
   disabling the client's timeout rather than inheriting it.
2. In the pinned `google-genai` 2.30.0, the SDK uses `HttpOptions.timeout` for both the local request
   timeout and, by default, `X-Server-Timeout`. However, `populate_server_timeout_header` preserves an
   already supplied `X-Server-Timeout` header. Independent values do not require an SDK change.
3. These two limits have different meanings: an httpx read timeout limits the wait between received
   chunks, while the server header requests a deadline for the whole request. Deriving the latter
   from the former can also constrain streams with an explicit 600-second read timeout; this is
   wider than detecting httpx's default five seconds.
4. The current provider maps a default five-second client to a five-second header; `read=None`
   becomes `DEFAULT_HTTP_TIMEOUT` (600 seconds). Provider-owned clients use a 20-minute timeout.
   Header construction is verified offline. Whether Vertex or Gemini actually aborts a request,
   and the precise response, still needs live confirmation. A local read timeout can fire first;
   the original claim that every generation aborts server-side with a 504 was too strong.
5. A returned 504 `DEADLINE_EXCEEDED` currently falls under `ModelServerError`. Changing that
   classification to `ModelTimeoutError` alone would not suppress default fallback:
   `FallbackModel` defaults to `(ModelAPIError,)`, which also includes `ModelTimeoutError` through
   `ModelConnectionError`. A different fallback policy would be a separate behavior decision.

## Offline request verification

Executed on 2026-10-31 with the PR's locked `google-genai==2.30.0`, httpx 0.28.1, an actual SDK
`aio.models.generate_content` call and `httpx.MockTransport`. The supplied client had a five-second
timeout. The transport captured the outgoing header and request timeout extensions; no Google
service or credentials were used.

| `HttpOptions.timeout` (ms) | Pre-set header | Observed `X-Server-Timeout` | Effective request read timeout |
| --- | --- | --- | --- |
| 5000 | absent | 5 | 5 seconds |
| 1200000 | absent | 1200 | 1200 seconds |
| 5000 | 1200 | 1200 | 5 seconds |

The second row disproves the original substitute-and-warn proposal's claim that replacing the SDK
timeout preserves the caller's local five-second timeout. The third confirms the pre-set-header
mechanism. These checks verify SDK request construction, not live server enforcement or elapsed time.

## Maintainer direction

Per [Douwe's comment](https://github.com/pydantic/pydantic-ai/issues/11902#issuecomment-6536168151):

- Keep pinning the local timeout to preserve the anti-hang behavior, and set the server deadline
  separately. The proposed server value is likely Google's 20-minute long-decode figure, or the
  read timeout if it is longer; the exact policy belongs to the implementation and its tests.
- Warn about a supplied client with the default five-second timeout, without overriding it: an
  explicit five-second choice is indistinguishable from the default.
- Preserve the distinction between a locally raised `ModelTimeoutError` and a 504 received from
  the server. Changing default fallback policy is not implied by that distinction.

This supersedes the original recommendation to substitute `HttpOptions.timeout` and seek an
upstream SDK change. No new public server-deadline parameter is proposed by this revision.

## Verification needed for the implementation

- Capture both outgoing headers and effective transport timeouts, rather than only `HttpOptions`.
  Cover provider-owned clients and supplied clients with default, explicit five-second, 600-second,
  longer-than-20-minute and `read=None` settings. Preserve the intentional finite timeout pin and
  check existing Gemini timeout constraints before choosing defaults.
- Check that streaming can continue beyond one read-timeout interval when chunks keep arriving,
  subject to the separately selected server deadline.
- Check local timeout classification and genuine server 504 classification separately. Do not
  assert that either bypasses fallback without an explicit fallback-policy change and regression test.
- Confirm live Vertex and Gemini API handling of `X-Server-Timeout` with someone who has access.
  Server enforcement, deadline defaults, and real 504 response bodies are not established by this
  offline probe. The maintainer has requested help obtaining this evidence.
