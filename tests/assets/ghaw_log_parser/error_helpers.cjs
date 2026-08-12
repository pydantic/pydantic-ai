// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/error_helpers.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check

/**
 * Detect whether a string looks like an HTML page rather than a plain error message.
 * Used to sanitize GitHub's "Unicorn" / gateway error HTML responses so they don't
 * pollute CI logs with hundreds of lines of markup.
 *
 * @param {string} str - The string to inspect
 * @returns {boolean} True when the string appears to be an HTML document
 */
function isHtmlContent(str) {
  return /^\s*<!DOCTYPE\s/i.test(str) || /^\s*<html[\s>]/i.test(str);
}

/**
 * Safely extract an error message from an unknown error value.
 * Handles Error instances, objects with message properties, and other values.
 *
 * When the extracted message looks like an HTML page (e.g. GitHub's "Unicorn"
 * 504 error page), it is replaced with a concise human-readable description so
 * that CI logs stay readable.  The HTTP status code is included when available.
 *
 * @param {unknown} error - The error value to extract a message from
 * @returns {string} The error message as a string
 */
function getErrorMessage(error) {
  // prettier-ignore
  const errorAsAny = /** @type {any} */ (error);
  let message;
  if (error instanceof Error) {
    message = error.message;
  } else if (error && typeof error === "object" && "message" in error && typeof error.message === "string") {
    message = error.message;
  } else {
    message = String(error);
  }

  if (isHtmlContent(message)) {
    const status = errorAsAny != null && typeof errorAsAny.status === "number" ? errorAsAny.status : null;
    return status != null ? `GitHub returned an unexpected HTML response (HTTP ${status})` : "GitHub returned an unexpected HTML response";
  }

  return message;
}

/**
 * Check if an error is due to a locked issue/PR/discussion.
 * GitHub API returns 403 with specific messages for locked resources.
 * This helper is used to determine if an operation should be silently ignored.
 *
 * @param {unknown} error - The error value to check
 * @returns {boolean} True if error is due to locked resource, false otherwise
 */
function isLockedError(error) {
  // Check if the error has a 403 status code
  const is403Error = error && typeof error === "object" && "status" in error && error.status === 403;
  if (!is403Error) {
    return false;
  }

  // Check if the error message mentions "locked"
  const errorMessage = getErrorMessage(error);
  const hasLockedMessage = Boolean(errorMessage && (errorMessage.includes("locked") || errorMessage.includes("Lock conversation")));

  // Only return true if it's BOTH a 403 status code AND mentions locked
  return hasLockedMessage;
}

/**
 * Check if an error is due to a GitHub API rate limit being exceeded.
 * This includes both installation-level and user-level rate limits.
 * Used to determine if a check should fail-open (allow workflow to proceed)
 * rather than hard-failing when the error is transient.
 *
 * @param {unknown} error - The error value to check
 * @returns {boolean} True if error is due to API rate limiting, false otherwise
 */
function isRateLimitError(error) {
  const errorMessage = getErrorMessage(error);
  return /\bapi rate limit\b|\brate limit exceeded\b/i.test(errorMessage);
}

module.exports = { getErrorMessage, isHtmlContent, isLockedError, isRateLimitError };
