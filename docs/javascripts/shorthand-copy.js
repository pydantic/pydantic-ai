/**
 * Make model shorthands (e.g., 'anthropic:claude-opus-4-5') clickable and copyable
 * This script finds inline code blocks on the popular-models page that contain
 * model shorthand IDs and adds copy-on-click functionality.
 */

function initializeShorthandCopy() {
  // Find all inline code elements
  const codeElements = document.querySelectorAll('code:not([data-shorthand-processed])');

  codeElements.forEach((code) => {
    const text = code.textContent.trim();

    // Pattern: matches model shorthands like 'anthropic:claude-opus-4-5' or 'gateway/anthropic:claude-opus-4-5'
    // Remove surrounding quotes if present
    const cleanText = text.replace(/^['"]|['"]$/g, '');

    // Check if it looks like a model shorthand (contains : or gateway/)
    if (
      cleanText.includes(':') &&
      (cleanText.startsWith('anthropic:') ||
        cleanText.startsWith('azure:') ||
        cleanText.startsWith('google-') ||
        cleanText.startsWith('openai:') ||
        cleanText.startsWith('grok:') ||
        cleanText.startsWith('gateway/') ||
        cleanText.startsWith('bedrock:') ||
        cleanText.startsWith('openrouter:') ||
        cleanText.startsWith('google-vertex:'))
    ) {
      // Mark as processed to avoid re-processing
      code.setAttribute('data-shorthand-processed', 'true');

      // Add copyable class for styling
      code.classList.add('shorthand-copyable');
      code.style.cursor = 'pointer';

      code.addEventListener('click', async (e) => {
        e.preventDefault();
        e.stopPropagation();

        // Copy to clipboard
        try {
          await navigator.clipboard.writeText(cleanText);

          // Show feedback
          const originalText = code.textContent;
          code.textContent = '✓ Copied!';
          code.classList.add('copied');

          setTimeout(() => {
            code.textContent = originalText;
            code.classList.remove('copied');
          }, 500);
        } catch (err) {
          console.error('Failed to copy:', err);
        }
      });
    }
  });
}

// Run when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeShorthandCopy);
} else {
  initializeShorthandCopy();
}

// Also run on any dynamic content updates (for search results, etc.)
const observer = new MutationObserver(() => {
  initializeShorthandCopy();
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
});
