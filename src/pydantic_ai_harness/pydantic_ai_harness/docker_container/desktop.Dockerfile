FROM debian:bookworm-slim
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xvfb xdotool x11-utils scrot fluxbox tint2 xterm firefox-esr \
        fonts-dejavu-core fonts-wqy-microhei fonts-noto-color-emoji ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/lib/firefox-esr/distribution /root/.fluxbox /root/.config/tint2 \
    && printf '%s\n' '{"policies": {"OverrideFirstRunPage": "", "OverridePostUpdatePage": "", "DontCheckDefaultBrowser": true, "DisableTelemetry": true, "DisplayBookmarksToolbar": "never", "UserMessaging": {"SkipOnboarding": true, "MoreFromMozilla": false, "FeatureRecommendations": false, "ExtensionRecommendations": false}, "FirefoxHome": {"SponsoredTopSites": false, "SponsoredPocket": false, "Pocket": false, "Highlights": false, "Snippets": false}, "Preferences": {"datareporting.policy.dataSubmissionPolicyBypassNotification": {"Value": true, "Status": "locked"}, "browser.newtabpage.activity-stream.showWeather": {"Value": false, "Status": "locked"}, "browser.sessionstore.resume_from_crash": {"Value": false, "Status": "locked"}}}}' > /usr/lib/firefox-esr/distribution/policies.json \
    && printf '%s\n' 'session.screen0.toolbar.visible: false' > /root/.fluxbox/init \
    && printf '%s\n' '[app] (class=firefox-esr)' '  [Maximized] {yes}' '[end]' \
        '[app] (class=XTerm)' '  [Maximized] {yes}' '[end]' > /root/.fluxbox/apps \
    && printf '%s\n' 'rounded = 0' 'border_width = 0' 'background_color = #2b2b2b 100' \
        'border_color = #2b2b2b 0' 'panel_items = LT' 'panel_position = bottom left horizontal' \
        'panel_size = 100% 40' 'panel_padding = 4 4 6' 'panel_background_id = 1' 'panel_layer = top' \
        'strut_policy = follow_size' 'launcher_icon_size = 32' 'launcher_padding = 2 4 8' \
        'launcher_item_app = /usr/share/applications/firefox-esr.desktop' \
        'launcher_item_app = /usr/share/applications/debian-xterm.desktop' \
        'taskbar_padding = 2 2 4' 'task_text = 1' 'task_icon = 1' 'task_font_color = #ffffff 100' \
        'task_maximum_size = 220 34' > /root/.config/tint2/tint2rc
# fluxbox's wallpaper helper finds no wallpaper setter and says so in an `xmessage` window on every start.
RUN rm -f /usr/bin/xmessage
ENV LANG=C.UTF-8
