## This check

The URL guard has repeatedly been fixed one instance at a time. Review the whole class at each changed site, not only the instance the diff names.

1. **Address forms the guard misses.** Every address check in `_ssrf.py` must handle IPv4-mapped, IPv4-compatible, 6to4, NAT64 (well-known and local-use prefixes) and ISATAP forms, and zone identifiers. The metadata address lists must cover each cloud provider's documented metadata endpoints.
2. **Domain-list semantics.** Compare the normalized hostname with normalized list entries: case, trailing dot, IDNA, and fullwidth or other compatibility forms. Apply the list before any DNS lookup, and again on every redirect.
3. **A limit applied after the expensive work.** Flag buffering a whole body before the size cap, or decoding, formatting or converting content before `max_content_length` or a similar cap truncates it.
4. **Super-linear or blocking processing of fetched content.** Check regular expressions, per-element or per-nesting-level reprocessing, JSON re-indenting, and charset detection and decoding, including work done on the event loop.
5. **A fetch or file reference that bypasses the guard.** Flag a new download path that does not use `safe_download`, a path that skips the reset of a client-supplied `force_download`, and an implicit server-side download of a URL. Flag a client-supplied `UploadedFile` or provider file identifier accepted without the adapter's gate.
6. **Browser capabilities.** In `playwright/` and `browser_use/`, check that navigation, each redirect and each subresource pass the configured address and domain checks.

A provider model file reaches this check when a changed line calls `download_item` or `safe_download`, or names `force_download`, `allow_local`, `UploadedFile(` or `_CLOUD_METADATA`. Review that call site and the URLs it can receive.
