// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/error_codes.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check

/**
 * Standardized error codes for safe-outputs handlers.
 *
 * These codes provide machine-readable prefixes for error messages,
 * enabling structured logging, monitoring dashboards, and alerting rules.
 *
 * Usage:
 *   const { ERR_VALIDATION } = require("./error_codes.cjs");
 *   throw new Error(`${ERR_VALIDATION}: Missing required field: title`);
 *   core.setFailed(`${ERR_CONFIG}: GH_AW_PROMPT environment variable is not set`);
 *
 * Error code categories:
 *   ERR_VALIDATION  - Input validation failures (missing fields, invalid format, limits exceeded)
 *   ERR_PERMISSION  - Authorization and permission check failures
 *   ERR_API         - GitHub API call failures
 *   ERR_CONFIG      - Configuration errors (missing env vars, bad setup)
 *   ERR_NOT_FOUND   - Resource not found (issues, discussions, PRs)
 *   ERR_PARSE       - Parsing failures (JSON, NDJSON, log formats)
 *   ERR_SYSTEM      - System and I/O errors (file access, git operations)
 */

/** @type {string} Input validation failures */
const ERR_VALIDATION = "ERR_VALIDATION";

/** @type {string} Authorization and permission check failures */
const ERR_PERMISSION = "ERR_PERMISSION";

/** @type {string} GitHub API call failures */
const ERR_API = "ERR_API";

/** @type {string} Configuration errors (missing env vars, bad setup) */
const ERR_CONFIG = "ERR_CONFIG";

/** @type {string} Resource not found */
const ERR_NOT_FOUND = "ERR_NOT_FOUND";

/** @type {string} Parsing failures (JSON, NDJSON, log formats) */
const ERR_PARSE = "ERR_PARSE";

/** @type {string} System and I/O errors */
const ERR_SYSTEM = "ERR_SYSTEM";

/** @type {string} Safe output validation/input errors (legacy numeric taxonomy) */
const SAFE_OUTPUT_E001 = "E001";

/** @type {string} Safe output lock file frontmatter hash mismatch (legacy numeric taxonomy) */
const SAFE_OUTPUT_E009 = "E009";

/** @type {string} Safe output retry exhaustion due to rate limiting (legacy numeric taxonomy) */
const SAFE_OUTPUT_E010 = "E010";

/** @type {string} Safe output operation/runtime failures (legacy numeric taxonomy) */
const SAFE_OUTPUT_E099 = "E099";

/** @type {string} Named code for lock file frontmatter hash mismatch */
const CONFIG_HASH_MISMATCH = "CONFIG_HASH_MISMATCH";

/** @type {string} Named code for rate limit retry exhaustion */
const RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED";

module.exports = {
  ERR_VALIDATION,
  ERR_PERMISSION,
  ERR_API,
  ERR_CONFIG,
  ERR_NOT_FOUND,
  ERR_PARSE,
  ERR_SYSTEM,
  SAFE_OUTPUT_E001,
  SAFE_OUTPUT_E009,
  SAFE_OUTPUT_E010,
  SAFE_OUTPUT_E099,
  CONFIG_HASH_MISMATCH,
  RATE_LIMIT_EXCEEDED,
};
