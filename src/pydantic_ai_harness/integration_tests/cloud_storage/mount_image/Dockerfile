# A sandbox image that can mount buckets: Python for the mount helper and fuse3 for `fusermount3`.
# rclone is left out on purpose, so the tests cover the helper installing it.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 fuse3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
