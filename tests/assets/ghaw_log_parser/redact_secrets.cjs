// Vendored verbatim (except this header) from github/gh-aw, MIT licensed — see LICENSE.
// Upstream: https://github.com/github/gh-aw/blob/fafef5837db7134eb1931954423f5c9d6e0bec3a/actions/setup/js/redact_secrets.cjs
// Pinned commit: fafef5837db7134eb1931954423f5c9d6e0bec3a
// @ts-check
/// <reference types="@actions/github-script" />

/**
 * Redacts secrets from files in /tmp/gh-aw and ${RUNNER_TEMP}/gh-aw directories before uploading artifacts
 * This script processes all .txt, .json, .log, .md, .mdx, .yml, .jsonl files under /tmp/gh-aw and ${RUNNER_TEMP}/gh-aw
 * and redacts any strings matching the actual secret values provided via environment variables.
 */
const fs = require("fs");
const path = require("path");
const { getErrorMessage } = require("./error_helpers.cjs");
const { ERR_VALIDATION } = require("./error_codes.cjs");
/**
 * Recursively finds all files matching the specified extensions
 * @param {string} dir - Directory to search
 * @param {string[]} extensions - File extensions to match (e.g., ['.txt', '.json', '.log'])
 * @returns {string[]} Array of file paths
 */
function findFiles(dir, extensions) {
  const results = [];
  try {
    if (!fs.existsSync(dir)) {
      return results;
    }
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // Recursively search subdirectories
        results.push(...findFiles(fullPath, extensions));
      } else if (entry.isFile()) {
        // Check if file has one of the target extensions
        const ext = path.extname(entry.name).toLowerCase();
        if (extensions.includes(ext)) {
          results.push(fullPath);
        }
      }
    }
  } catch (error) {
    core.warning(`Failed to scan directory ${dir}: ${getErrorMessage(error)}`);
  }
  return results;
}

/**
 * Built-in regex patterns for common credential types
 * Each pattern is designed to match legitimate credential formats
 */
const BUILT_IN_PATTERNS = [
  // GitHub tokens
  { name: "GitHub Personal Access Token (classic)", pattern: /ghp_[0-9a-zA-Z]{36}/g },
  // New stateless ghs_ token format allows dots, underscores, and dashes
  // and uses variable length (minimum 36 chars after prefix).
  // https://github.blog/changelog/2026-05-15-github-app-installation-tokens-per-request-override-header/
  { name: "GitHub Server-to-Server Token", pattern: /ghs_[0-9A-Za-z._-]{36,}/g },
  { name: "GitHub OAuth Access Token", pattern: /gho_[0-9a-zA-Z]{36}/g },
  { name: "GitHub User Access Token", pattern: /ghu_[0-9a-zA-Z]{36}/g },
  { name: "GitHub Fine-grained PAT", pattern: /github_pat_[0-9a-zA-Z_]{82}/g },
  { name: "GitHub Refresh Token", pattern: /ghr_[0-9a-zA-Z]{36}/g },

  // Azure tokens
  { name: "Azure Storage Account Key", pattern: /AccountKey=[a-zA-Z0-9+/]{86}==/g },
  { name: "Azure SAS Token", pattern: /\?sv=[0-9-]{1,20}&s[rts]=[\w\-]{1,20}&sig=[A-Za-z0-9%+/=]{1,200}/g },

  // Google/GCP tokens
  { name: "Google API Key", pattern: /AIzaSy[0-9A-Za-z_-]{33}/g },
  { name: "Google OAuth Access Token", pattern: /ya29\.[0-9A-Za-z_-]{1,800}/g },

  // AWS tokens
  { name: "AWS Access Key ID", pattern: /AKIA[0-9A-Z]{16}/g },

  // OpenAI tokens
  { name: "OpenAI API Key", pattern: /sk-[a-zA-Z0-9]{48}/g },
  { name: "OpenAI Project API Key", pattern: /sk-proj-[a-zA-Z0-9]{48,64}/g },

  // Anthropic tokens
  { name: "Anthropic API Key", pattern: /sk-ant-api03-[a-zA-Z0-9_-]{95}/g },
];

/**
 * MCP gateway configuration files that may contain bearer tokens.
 * These are the canonical paths produced by the gateway setup scripts.
 * The list is defined as a module-level constant so tests can replace entries.
 */
// Shell setup scripts write under /tmp, while CJS converters use RUNNER_TEMP.
const MCP_GATEWAY_CONFIG_PATHS = [
  ...new Set(
    [
      path.join("/tmp", "gh-aw/mcp-config/gateway-output.json"),
      path.join("/tmp", "gh-aw/mcp-config/mcp-servers.json"),
      path.join("/tmp", "gh-aw/mcp-config/config.toml"),
      path.join(process.env.RUNNER_TEMP || "/tmp", "gh-aw/mcp-config/gateway-output.json"),
      path.join(process.env.RUNNER_TEMP || "/tmp", "gh-aw/mcp-config/mcp-servers.json"),
      path.join(process.env.RUNNER_TEMP || "/tmp", "gh-aw/mcp-config/config.toml"),
      process.env.HOME ? path.join(process.env.HOME, ".copilot/mcp-config.json") : "",
      process.env.GITHUB_WORKSPACE ? path.join(process.env.GITHUB_WORKSPACE, ".gemini/settings.json") : "",
    ].filter(Boolean)
  ),
];

/**
 * Minimum credential length required before an Authorization value is treated
 * as a gateway token. Guards against redacting short placeholder values.
 */
const MIN_GATEWAY_TOKEN_LENGTH = 6;

/**
 * Extracts MCP gateway bearer tokens from known configuration files.
 * The gateway token is dynamically minted and has no recognisable prefix,
 * so it cannot be caught by the built-in regex patterns.  We read the
 * gateway config files directly and treat every Authorization header value
 * as a secret to be redacted.
 * @param {string[]} configPaths - Paths to MCP gateway config JSON files
 * @returns {string[]} Unique token values extracted from the files
 */
function extractMCPGatewayTokens(configPaths) {
  /** @type {Set<string>} */
  const tokens = new Set();

  /**
   * Records an Authorization header value plus, for a bearer header, the bare
   * credential so the token is redacted even when logged without the prefix.
   * @param {unknown} value - Raw Authorization header value
   */
  const addAuthorizationValue = value => {
    if (typeof value !== "string") return;
    const trimmed = value.trim();
    const bearerMatch = /^[Bb]earer\s+(.+)$/.exec(trimmed);
    // The minimum-length guard applies to the credential itself, never to the
    // bearer prefix, so short values cannot slip through by being prefixed.
    const credential = bearerMatch ? bearerMatch[1].trim() : trimmed;
    if (credential.length < MIN_GATEWAY_TOKEN_LENGTH) return;
    tokens.add(trimmed);
    tokens.add(credential);
  };

  for (const configPath of configPaths) {
    try {
      if (!fs.existsSync(configPath)) continue;
      const raw = fs.readFileSync(configPath, "utf8");
      let config;
      try {
        config = /** @type {Record<string, any>} */ JSON.parse(raw);
      } catch {
        // Codex writes TOML (`http_headers = { Authorization = "..." }`); the key
        // is matched case-insensitively to tolerate formatting differences.
        for (const match of raw.matchAll(/\bAuthorization\s*=\s*"([^"]+)"/gi)) {
          addAuthorizationValue(match[1]);
        }
        continue;
      }
      const servers = /** @type {Record<string, any>} */ config.mcpServers || {};
      for (const server of Object.values(servers)) {
        addAuthorizationValue(server?.headers?.Authorization);
      }
    } catch {
      // Silently skip unreadable or malformed files — absence of the gateway
      // config is normal when the MCP gateway is not used by a workflow.
    }
  }
  return [...tokens];
}

/**
 * Detects and redacts secrets matching built-in patterns
 * @param {string} content - File content to process
 * @returns {{content: string, redactionCount: number, detectedPatterns: string[]}} Redacted content, count, and detected pattern types
 */
function redactBuiltInPatterns(content) {
  let redactionCount = 0;
  let redacted = content;
  const detectedPatterns = [];

  for (const { name, pattern } of BUILT_IN_PATTERNS) {
    const matches = redacted.match(pattern);
    if (matches && matches.length > 0) {
      // Redact each match with fixed-length string
      const replacement = "***REDACTED***";
      for (const match of matches) {
        redacted = redacted.split(match).join(replacement);
      }
      redactionCount += matches.length;
      detectedPatterns.push(name);
      core.info(`Redacted ${matches.length} occurrence(s) of ${name}`);
    }
  }

  return { content: redacted, redactionCount, detectedPatterns };
}

/**
 * Redacts secrets from file content using exact string matching
 * @param {string} content - File content to process
 * @param {string[]} secretValues - Array of secret values to redact
 * @returns {{content: string, redactionCount: number}} Redacted content and count of redactions
 */
function redactSecrets(content, secretValues) {
  let redactionCount = 0;
  let redacted = content;
  // Sort secret values by length (longest first) to handle overlapping secrets
  const sortedSecrets = secretValues.slice().sort((a, b) => b.length - a.length);
  for (const secretValue of sortedSecrets) {
    // Skip empty or very short values (likely not actual secrets)
    if (!secretValue || secretValue.length < 6) {
      continue;
    }
    // Count occurrences before replacement
    // Use split and join for exact string matching (not regex)
    // This is safer than regex as it doesn't interpret special characters
    // Use fixed-length redaction string without prefix preservation
    const replacement = "***REDACTED***";
    const parts = redacted.split(secretValue);
    const occurrences = parts.length - 1;
    if (occurrences > 0) {
      redacted = parts.join(replacement);
      redactionCount += occurrences;
      core.info(`Redacted ${occurrences} occurrence(s) of a secret`);
    }
  }
  return { content: redacted, redactionCount };
}

/**
 * Redacts credential-shaped strings from content destined for the GitHub Actions
 * step summary.
 *
 * Step summaries reproduce agent-controlled data (tool inputs, tool outputs, agent
 * text, safe-output titles), and `::add-mask::` processing does not scrub
 * `$GITHUB_STEP_SUMMARY`, so the built-in credential patterns used for artifact
 * redaction are applied before the summary is written. Redaction failures are
 * non-fatal because the summary is best-effort output.
 *
 * @param {string} content - Markdown destined for the step summary
 * @returns {string} Content with credential-shaped strings replaced
 */
function redactStepSummaryContent(content) {
  if (typeof content !== "string" || content.length === 0) {
    return content;
  }
  try {
    return redactBuiltInPatterns(content).content;
  } catch (error) {
    core.warning(`Failed to redact step summary content: ${getErrorMessage(error)}`);
    return content;
  }
}

/**
 * Process a single file for secret redaction
 * @param {string} filePath - Path to the file
 * @param {string[]} secretValues - Array of secret values to redact
 * @returns {number} Number of redactions made
 */
function processFile(filePath, secretValues) {
  try {
    const content = fs.readFileSync(filePath, "utf8");

    // First, redact built-in patterns
    const builtInResult = redactBuiltInPatterns(content);
    let redacted = builtInResult.content;
    let totalRedactions = builtInResult.redactionCount;

    // Then, redact custom secrets
    const customResult = redactSecrets(redacted, secretValues);
    redacted = customResult.content;
    totalRedactions += customResult.redactionCount;

    if (totalRedactions > 0) {
      fs.writeFileSync(filePath, redacted, "utf8");
      core.info(`Processed ${filePath}: ${totalRedactions} redaction(s)`);
    }
    return totalRedactions;
  } catch (error) {
    core.warning(`Failed to process file ${filePath}: ${getErrorMessage(error)}`);
    return 0;
  }
}

/**
 * Main function
 */
async function main() {
  // Get the list of secret names from environment variable
  const secretNames = process.env.GH_AW_SECRET_NAMES;

  core.info(`Starting secret redaction in /tmp/gh-aw and ${process.env.RUNNER_TEMP}/gh-aw directories`);
  try {
    // Collect custom secret values from environment variables
    const secretValues = [];
    if (secretNames) {
      // Parse the comma-separated list of secret names
      const secretNameList = secretNames.split(",").filter(name => name.trim());
      for (const secretName of secretNameList) {
        const envVarName = `SECRET_${secretName}`;
        const secretValue = process.env[envVarName];
        // Skip empty or undefined secrets
        if (!secretValue || secretValue.trim() === "") {
          continue;
        }
        secretValues.push(secretValue.trim());
      }
    }

    if (secretValues.length > 0) {
      core.info(`Found ${secretValues.length} custom secret(s) to redact`);
    }

    // Extract MCP gateway bearer tokens from known config files and add them to
    // the redaction list.  The gateway token has no fixed prefix so it cannot be
    // matched by the built-in regex patterns; we must read it from the config.
    const gatewayTokens = extractMCPGatewayTokens(MCP_GATEWAY_CONFIG_PATHS);
    if (gatewayTokens.length > 0) {
      core.info(`Found ${gatewayTokens.length} MCP gateway token(s) to redact`);
      secretValues.push(...gatewayTokens);
    }

    // Always scan for built-in patterns, even if there are no custom secrets
    core.info("Scanning for built-in credential patterns and custom secrets");

    // Find all target files in /tmp/gh-aw and ${RUNNER_TEMP}/gh-aw directories
    const targetExtensions = [".txt", ".json", ".log", ".md", ".mdx", ".yml", ".jsonl", ".patch"];
    const tmpFiles = findFiles("/tmp/gh-aw", targetExtensions);
    const optFiles = findFiles(`${process.env.RUNNER_TEMP}/gh-aw`, targetExtensions);
    const files = [...tmpFiles, ...optFiles];
    core.info(`Found ${files.length} file(s) to scan for secrets (${tmpFiles.length} in /tmp/gh-aw, ${optFiles.length} in ${process.env.RUNNER_TEMP}/gh-aw)`);
    let totalRedactions = 0;
    let filesWithRedactions = 0;
    // Process each file
    for (const file of files) {
      const redactionCount = processFile(file, secretValues);
      if (redactionCount > 0) {
        filesWithRedactions++;
        totalRedactions += redactionCount;
      }
    }
    if (totalRedactions > 0) {
      core.info(`Secret redaction complete: ${totalRedactions} redaction(s) in ${filesWithRedactions} file(s)`);
    } else {
      core.info("Secret redaction complete: no secrets found");
    }
  } catch (error) {
    core.setFailed(`${ERR_VALIDATION}: Secret redaction failed: ${getErrorMessage(error)}`);
  }
}

module.exports = { main, redactSecrets, redactBuiltInPatterns, redactStepSummaryContent, extractMCPGatewayTokens, BUILT_IN_PATTERNS, MCP_GATEWAY_CONFIG_PATHS };
