// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/markdown_unfencing.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check

/**
 * Markdown Unfencing Utility
 *
 * Removes outer code fences from markdown content when the entire content
 * is wrapped in a markdown/md code fence. This handles cases where agents
 * accidentally wrap the entire markdown body in a code fence.
 */

/**
 * Unfence markdown content by removing outer code fence if present.
 *
 * The function detects:
 * - Content starting with ```markdown, ```md, ~~~markdown, or ~~~md (case insensitive)
 * - Content ending with ``` or ~~~
 * - The closing fence must match the opening fence type (backticks or tildes)
 *
 * @param {string} content - The markdown content to unfence
 * @returns {string} The unfenced content if a wrapping fence is detected, otherwise original content
 */
function unfenceMarkdown(content) {
  if (!content || typeof content !== "string") {
    return content;
  }

  // Trim leading/trailing whitespace for analysis
  const trimmed = content.trim();

  // Split into lines
  const lines = trimmed.split("\n");
  if (lines.length < 2) {
    // Need at least opening fence and closing fence
    return content;
  }

  const firstLine = lines[0].trim();
  const lastLine = lines[lines.length - 1].trim();

  // Check if first line is a markdown code fence
  /** @type {any} */
  let fenceChar = null;
  let fenceLength = 0;
  let isMarkdownFence = false;

  // Check for backtick fences (3 or more backticks)
  if (firstLine.startsWith("```")) {
    fenceChar = "`";
    // Count the number of consecutive backticks
    for (const ch of firstLine) {
      if (ch === "`") {
        fenceLength++;
      } else {
        break;
      }
    }
    const remainder = firstLine.substring(fenceLength).trim();
    // Check if it's markdown or md language tag or empty
    if (remainder === "" || remainder.toLowerCase() === "markdown" || remainder.toLowerCase() === "md") {
      isMarkdownFence = true;
    }
  } else if (firstLine.startsWith("~~~")) {
    // Check for tilde fences (3 or more tildes)
    fenceChar = "~";
    // Count the number of consecutive tildes
    for (const ch of firstLine) {
      if (ch === "~") {
        fenceLength++;
      } else {
        break;
      }
    }
    const remainder = firstLine.substring(fenceLength).trim();
    // Check if it's markdown or md language tag or empty
    if (remainder === "" || remainder.toLowerCase() === "markdown" || remainder.toLowerCase() === "md") {
      isMarkdownFence = true;
    }
  }

  if (!isMarkdownFence) {
    // Not a markdown fence, return original content
    return content;
  }

  // Check if last line is a matching closing fence
  // Must have at least as many fence characters as the opening fence
  let isClosingFence = false;
  if (fenceChar === "`") {
    // Count backticks in last line
    let closingFenceLength = 0;
    for (const ch of lastLine) {
      if (ch === "`") {
        closingFenceLength++;
      } else {
        break;
      }
    }
    // Must have at least as many backticks as opening fence
    if (closingFenceLength >= fenceLength && lastLine.substring(closingFenceLength).trim() === "") {
      isClosingFence = true;
    }
  } else if (fenceChar === "~") {
    // Count tildes in last line
    let closingFenceLength = 0;
    for (const ch of lastLine) {
      if (ch === "~") {
        closingFenceLength++;
      } else {
        break;
      }
    }
    // Must have at least as many tildes as opening fence
    if (closingFenceLength >= fenceLength && lastLine.substring(closingFenceLength).trim() === "") {
      isClosingFence = true;
    }
  }

  if (!isClosingFence) {
    // No matching closing fence, return original content
    return content;
  }

  // Extract the content between the fences
  // Remove first and last lines
  const innerLines = lines.slice(1, -1);
  const innerContent = innerLines.join("\n");

  if (typeof core !== "undefined") {
    core.info(`Unfenced markdown content: removed outer ${fenceChar} fence`);
  }

  // Return the inner content with original leading/trailing whitespace style preserved
  return innerContent.trim();
}

module.exports = {
  unfenceMarkdown,
};
