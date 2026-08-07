// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/log_parser_step_summary_builder.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check

/**
 * Builds a step-summary section with a collapsible details body whose summary
 * acts as the section header.
 * @param {string} title
 * @param {string} body
 * @param {{open?: boolean, emptyBodyMessage?: string}} [options]
 * @returns {string}
 */
function buildStepSummaryDetailsSection(title, body, options = {}) {
  const { open = false, emptyBodyMessage = "No details available." } = options;
  const openAttr = open ? " open" : "";
  const trimmedBody = typeof body === "string" ? body.trim() : "";
  const content = trimmedBody || emptyBodyMessage;
  return `<details${openAttr}>\n<summary>${title}</summary>\n\n${content}\n</details>\n\n`;
}

module.exports = { buildStepSummaryDetailsSection };
