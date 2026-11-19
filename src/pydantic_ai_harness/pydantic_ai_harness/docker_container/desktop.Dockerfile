FROM debian:bookworm-slim
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xvfb xdotool x11-utils scrot fluxbox tint2 xterm firefox-esr \
        fonts-dejavu-core fonts-wqy-microhei fonts-noto-color-emoji ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# The window manager's and panel's settings are written when the display starts, as for every other sandbox.
RUN mkdir -p /usr/lib/firefox-esr/distribution \
    && printf '%s\n' '{"policies": {"OverrideFirstRunPage": "", "OverridePostUpdatePage": "", "DontCheckDefaultBrowser": true, "DisableTelemetry": true, "DisplayBookmarksToolbar": "never", "UserMessaging": {"SkipOnboarding": true, "MoreFromMozilla": false, "FeatureRecommendations": false, "ExtensionRecommendations": false}, "FirefoxHome": {"SponsoredTopSites": false, "SponsoredPocket": false, "Pocket": false, "Highlights": false, "Snippets": false}, "Preferences": {"datareporting.policy.dataSubmissionPolicyBypassNotification": {"Value": true, "Status": "locked"}, "browser.newtabpage.activity-stream.showWeather": {"Value": false, "Status": "locked"}, "browser.sessionstore.resume_from_crash": {"Value": false, "Status": "locked"}}}}' > /usr/lib/firefox-esr/distribution/policies.json
ENV LANG=C.UTF-8
