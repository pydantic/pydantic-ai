FROM debian:bookworm-slim
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        xvfb xdotool x11-utils scrot util-linux fluxbox tint2 xterm firefox-esr \
        fonts-dejavu-core fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=C.UTF-8
