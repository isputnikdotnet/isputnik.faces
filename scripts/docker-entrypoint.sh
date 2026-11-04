#!/bin/sh
# Run the server unprivileged. The image starts as root only long enough to drop
# privileges with gosu; this engine mounts nothing and writes nothing, so there
# is no volume to take ownership of. PUID/PGID choose the user: default 1000
# (the image's `node` user); on Unraid 99:100 matches the other containers.
# Started with a compose `user:` instead (already unprivileged), it runs as is.
set -e

PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

if [ "$(id -u)" = "0" ]; then
  exec gosu "${PUID}:${PGID}" "$@"
fi

exec "$@"
