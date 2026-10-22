"""Real-browser regressions run once in the dedicated CLAI 2 web CI job."""

import os

collect_ignore = ['test_web_browser.py'] if os.environ.get('CLAI2_BROWSER_TESTS') != '1' else []
