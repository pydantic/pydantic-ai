// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/parse_claude_log.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check
/// <reference types="@actions/github-script" />

const {
  createEngineLogParser,
  generateConversationMarkdown,
  generateInformationSection,
  buildStepSummaryDetailsSection,
  formatInitializationSummary,
  formatToolUse,
  parseLogEntries,
  convertLegacyLogEntriesToCopilotEvents,
} = require("./log_parser_shared.cjs");

const main = createEngineLogParser({
  parserName: "Claude",
  parseFunction: parseClaudeLog,
  supportsDirectories: false,
});

/**
 * Parses Claude log content and converts it to markdown format
 * @param {string} logContent - The raw log content as a string
 * @returns {{markdown: string, mcpFailures: string[], maxTurnsHit: boolean, logEntries: Array}} Result with formatted markdown content, MCP failure list, max-turns status, and parsed log entries
 */
function parseClaudeLog(logContent) {
  // Use shared parseLogEntries function
  const logEntries = parseLogEntries(logContent);

  if (!logEntries) {
    return {
      markdown: buildStepSummaryDetailsSection("Agent Log Summary", "Log format not recognized as Claude JSON array or JSONL."),
      mcpFailures: [],
      maxTurnsHit: false,
      logEntries: [],
    };
  }

  const mcpFailures = [];

  // Generate conversation markdown using shared function
  const canonicalLogEntries = convertLegacyLogEntriesToCopilotEvents(logEntries, { sourceEngine: "claude" });
  const conversationResult = generateConversationMarkdown(canonicalLogEntries, {
    formatToolCallback: (toolUse, toolResult) => formatToolUse(toolUse, toolResult, { includeDetailedParameters: false }),
    formatInitCallback: initEntry => {
      const result = formatInitializationSummary(initEntry, {
        includeSlashCommands: true,
        mcpFailureCallback: server => {
          // Display detailed error information for failed MCP servers (Claude-specific)
          const errorDetails = [];

          if (server.error) {
            errorDetails.push(`**Error:** ${server.error}`);
          }

          if (server.stderr) {
            // Truncate stderr if too long
            const maxStderrLength = 500;
            const stderr = server.stderr.length > maxStderrLength ? server.stderr.substring(0, maxStderrLength) + "..." : server.stderr;
            errorDetails.push(`**Stderr:** \`${stderr}\``);
          }

          if (server.exitCode !== undefined && server.exitCode !== null) {
            errorDetails.push(`**Exit Code:** ${server.exitCode}`);
          }

          if (server.command) {
            errorDetails.push(`**Command:** \`${server.command}\``);
          }

          if (server.message) {
            errorDetails.push(`**Message:** ${server.message}`);
          }

          if (server.reason) {
            errorDetails.push(`**Reason:** ${server.reason}`);
          }

          // Return formatted error details with proper indentation
          if (errorDetails.length > 0) {
            return errorDetails.map(detail => `  - ${detail}\n`).join("");
          }
          return "";
        },
      });

      // Track MCP failures
      if (result.mcpFailures) {
        mcpFailures.push(...result.mcpFailures);
      }
      return result;
    },
  });

  let markdown = conversationResult.markdown;

  // Add Information section from the last entry with result metadata
  const lastEntry = logEntries[logEntries.length - 1];
  markdown += generateInformationSection(lastEntry);

  // Check if max-turns limit was hit
  let maxTurnsHit = false;
  const maxTurns = process.env.GH_AW_MAX_TURNS;
  if (maxTurns && lastEntry && lastEntry.num_turns) {
    const configuredMaxTurns = parseInt(maxTurns, 10);
    if (!Number.isNaN(configuredMaxTurns) && lastEntry.num_turns >= configuredMaxTurns) {
      maxTurnsHit = true;
    }
  }

  return { markdown, mcpFailures, maxTurnsHit, logEntries: canonicalLogEntries };
}

// Export for testing
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    main,
    parseClaudeLog,
  };
}
